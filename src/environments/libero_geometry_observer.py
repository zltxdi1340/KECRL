"""Read-only, privileged geometry on an independent MuJoCo position pipeline.

Never refresh the live simulator or expose geometry to the Policy. Contacts
here are geometric boundary proxies, not solver forces or grasp qualification.
"""
from __future__ import annotations

import hashlib
import json
import math

import numpy as np


COPY_FIELDS = ('qpos', 'qvel', 'time', 'mocap_pos', 'mocap_quat', 'eq_active')


def require(ok, message):
    if not ok:
        raise ValueError(message)


def fingerprint(data, fields=None):
    """Hash numeric live buffers, including contact and diagnostic struct views."""
    h = hashlib.sha256()
    names = fields if fields is not None else [n for n in dir(data) if not n.startswith('_')]
    for name in sorted(names):
        value = getattr(data, name)
        if isinstance(value, np.ndarray):
            require(not value.dtype.hasobject, 'opaque object array in native data')
            h.update(name.encode()); h.update(str(value.dtype).encode())
            h.update(str(value.shape).encode())
            # In 3.2.3 inactive NULL arena pointers can surface as freshly
            # allocated, uninitialized NumPy arrays. They are not native
            # buffers. Hash their storage status, never their undefined bytes.
            if value.size and value.flags.owndata and value.base is None:
                h.update(b'NO_NATIVE_STORAGE')
            else:
                h.update(np.ascontiguousarray(value).tobytes())
        elif isinstance(value, (int, float, np.number)):
            h.update(name.encode()); h.update(repr(value).encode())
    if fields is None:
        for group in ('contact', 'warning', 'timer', 'solver'):
            value = getattr(data, group, None)
            if value is not None:
                h.update(group.encode()); h.update(fingerprint(value, () if callable(value) else
                    [n for n in dir(value) if not n.startswith('_')]).encode())
    return h.hexdigest()


def _vector(value, shape):
    array = np.asarray(value, np.float64)
    require(array.shape == shape and np.isfinite(array).all(), 'invalid geometry dimensions/values')
    return array.tolist()


def _rotation(value):
    r = np.asarray(value, np.float64).reshape(3, 3)
    require(np.isfinite(r).all() and np.allclose(r @ r.T, np.eye(3), atol=1e-6, rtol=0) and
            abs(float(np.linalg.det(r))-1) <= 1e-6, 'invalid world rotation')
    return r.tolist()


class LiberoGeometryObserver:
    def __init__(self, domain, *, observation_digest, ledger):
        import mujoco
        self.mujoco, self.domain = mujoco, domain
        self.model, self.native = domain.sim.model._model, domain.sim.data._data
        require(self.model.nplugin == 0, 'plugin-dependent geometry is outside this observer implementation')
        self.observation_digest, self.ledger = observation_digest, ledger
        require(len(domain.robots) == 1, 'one Panda required')
        robot = domain.robots[0]
        self.joint_indices = list(robot._ref_joint_pos_indexes)
        self.gripper_indices = list(robot._ref_gripper_joint_pos_indexes)

        def resolve(kind, name):
            index = mujoco.mj_name2id(self.model, kind, name)
            require(index >= 0 and mujoco.mj_id2name(self.model, kind, index) == name,
                    'missing or ambiguous model name: '+name)
            return int(index)

        site_name = robot.gripper.important_sites['grip_site']
        self.site = resolve(mujoco.mjtObj.mjOBJ_SITE, site_name)
        body_info, groups = {}, {}
        for key, name in [('bowl', 'akita_black_bowl_1'), ('plate', 'plate_1')]:
            obj = domain.objects_dict[name]
            body = resolve(mujoco.mjtObj.mjOBJ_BODY, obj.root_body)
            require(body == domain.obj_body_id[name], 'body ID disagrees with LIBERO mapping')
            body_info[key] = {'object_name': name, 'body_name': obj.root_body, 'body_id': body}
            groups[key] = [(n, resolve(mujoco.mjtObj.mjOBJ_GEOM, n)) for n in obj.contact_geoms]
        for key, group in [('left', 'left_finger'), ('right', 'right_finger')]:
            names = robot.gripper.important_geoms[group]
            require(len(names) == len(set(names)), 'duplicate finger collision names')
            groups[key] = [(n, resolve(mujoco.mjtObj.mjOBJ_GEOM, n)) for n in names]
        self.groups = {k: {i for _, i in rows} for k, rows in groups.items()}
        require(all(self.groups.values()) and all(len(groups[k]) == len(self.groups[k]) for k in groups),
                'empty or duplicate collision group')
        keys = list(self.groups)
        require(all(not self.groups[a] & self.groups[b] for i, a in enumerate(keys) for b in keys[i+1:]),
                'overlapping finger/object collision groups')
        require(abs(float(self.model.opt.timestep)-.002) <= 1e-15 and
                domain.control_freq == 20 and abs(domain.model_timestep-.002) <= 1e-15,
                'native control/physics cadence differs')
        self.body_info = body_info
        self.header = {
            'coordinate_frame': 'world', 'distance_unit': 'metre', 'rotation_format': '3x3_row_major',
            'position_origin': 'site_and_root_body_origins_not_COM',
            'contact_origin': 'shadow_mj_fwdPosition_geometric_collision_not_force',
            'fingerprint_scope': 'native_backed_numeric_buffers; inactive_NULL_pointer_numpy_allocations_marked_not_hashed',
            'eef_site_name': site_name, 'eef_site_id': self.site, 'bodies': body_info,
            'collision_groups': {k: [{'name': n, 'id': i} for n, i in rows] for k, rows in groups.items()},
            'joint_qpos_indices': [int(i) for i in self.joint_indices],
            'gripper_qpos_indices': [int(i) for i in self.gripper_indices],
            'model_xml_sha256': hashlib.sha256(domain.sim.model.get_xml().encode()).hexdigest(),
            'model_nq': self.model.nq, 'model_nv': self.model.nv,
            'model_nbody': self.model.nbody, 'model_ngeom': self.model.ngeom, 'model_nsite': self.model.nsite,
            'policy_proprioception_clock': 'original_sampled_cache; not refreshed to native boundary',
            'physics_timestep_seconds': float(self.model.opt.timestep), 'control_frequency_hz': domain.control_freq,
        }
        self.shadow = ledger.call('shadow_allocation', mujoco.MjData, self.model)
        for name in COPY_FIELDS:
            a, b = getattr(self.native, name), getattr(self.shadow, name)
            if isinstance(a, np.ndarray):
                require(a.shape == b.shape and not np.shares_memory(a, b) and
                        (a.size == 0 or not a.flags.owndata), 'native/shadow arrays alias or lack real storage')
        self.last_time, self.start_bowl_z = None, None

    def sample(self, observation, state_index):
        before, obs_before = fingerprint(self.native), self.observation_digest(observation)
        joint_gap = np.asarray(self.native.qpos[self.joint_indices], np.float64)-observation['joint_states']
        gripper_gap = np.asarray(self.native.qpos[self.gripper_indices], np.float64)-observation['gripper_states']
        require(joint_gap.shape == (7,) and gripper_gap.shape == (2,) and
                np.isfinite(joint_gap).all() and np.isfinite(gripper_gap).all(), 'invalid current/cached proprioception')
        for name in COPY_FIELDS:
            value = getattr(self.native, name)
            if isinstance(value, np.ndarray):
                getattr(self.shadow, name)[:] = value
            else:
                setattr(self.shadow, name, value)
        copied = fingerprint(self.native, COPY_FIELDS)
        require(copied == fingerprint(self.shadow, COPY_FIELDS), 'shadow copy differs')
        self.ledger.call('shadow_forward', self.mujoco.mj_fwdPosition, self.model, self.shadow)
        require(copied == fingerprint(self.shadow, COPY_FIELDS), 'position pipeline modified copied state')
        now = float(self.shadow.time)
        require(math.isfinite(now) and (self.last_time is None or abs(now-self.last_time-.05) <= 1e-9),
                'geometry control-boundary time differs')
        eef = np.asarray(self.shadow.site_xpos[self.site], np.float64)
        bowl_id, plate_id = [self.body_info[k]['body_id'] for k in ('bowl', 'plate')]
        bowl, plate = self.shadow.xpos[bowl_id], self.shadow.xpos[plate_id]
        contacts = []
        counts = dict.fromkeys(('left', 'right', 'plate', 'other'), 0)
        margin_counts = dict(counts)
        for i in range(self.shadow.ncon):
            contact = self.shadow.contact[i]
            g1, g2 = map(int, contact.geom)
            if not ({g1, g2} & self.groups['bowl']):
                continue
            require(g1 >= 0 and g2 >= 0, 'flex/unknown contact unsupported')
            other = g2 if g1 in self.groups['bowl'] else g1
            tag = next((key for key in ('left', 'right', 'plate') if other in self.groups[key]), 'other')
            distance = float(contact.dist)
            require(math.isfinite(distance), 'invalid contact distance')
            (counts if distance <= 0 else margin_counts)[tag] += 1
            contacts.append({'geom1': g1, 'geom2': g2,
                'geom1_name': self.mujoco.mj_id2name(self.model, self.mujoco.mjtObj.mjOBJ_GEOM, g1),
                'geom2_name': self.mujoco.mj_id2name(self.model, self.mujoco.mjtObj.mjOBJ_GEOM, g2),
                'other_group': tag, 'distance_m': distance, 'position_world': _vector(contact.pos, (3,))})
        require(len(contacts) <= 256, 'bowl contact count exceeds frozen cap; no truncation')
        contacts.sort(key=lambda r: (r['geom1'], r['geom2'], r['distance_m'], r['position_world']))
        if contacts:
            self.ledger.begin('contact_record', len(contacts)); self.ledger.finish('contact_record', len(contacts))
        if self.start_bowl_z is None:
            self.start_bowl_z = float(bowl[2])
        delta, target = bowl-eef, plate-bowl
        row = {'state_index': state_index, 'sim_time': now,
            'native_vs_policy_joint_gap_l2': float(np.linalg.norm(joint_gap)),
            'native_vs_policy_gripper_gap_l2': float(np.linalg.norm(gripper_gap)),
            'eef_position_world': _vector(eef, (3,)), 'eef_rotation_world': _rotation(self.shadow.site_xmat[self.site]),
            'bowl_position_world': _vector(bowl, (3,)), 'bowl_rotation_world': _rotation(self.shadow.xmat[bowl_id]),
            'plate_position_world': _vector(plate, (3,)), 'plate_rotation_world': _rotation(self.shadow.xmat[plate_id]),
            'eef_to_bowl_world': _vector(delta, (3,)), 'eef_to_bowl_xyz': float(np.linalg.norm(delta)),
            'eef_to_bowl_xy': float(np.linalg.norm(delta[:2])), 'eef_minus_bowl_signed_z': float(eef[2]-bowl[2]),
            'bowl_to_plate_world': _vector(target, (3,)), 'bowl_to_plate_xyz': float(np.linalg.norm(target)),
            'bowl_to_plate_xy': float(np.linalg.norm(target[:2])), 'bowl_height_change': float(bowl[2]-self.start_bowl_z),
            'contacts': contacts, 'touch_counts': counts, 'positive_margin_counts': margin_counts,
            'bilateral_touch_proxy': counts['left'] > 0 and counts['right'] > 0,
            'native_before_sha256': before, 'native_after_sha256': fingerprint(self.native),
            'observation_before_sha256': obs_before, 'observation_after_sha256': self.observation_digest(observation),
            'copied_state_sha256': copied, 'shadow_copied_state_sha256': fingerprint(self.shadow, COPY_FIELDS),
        }
        require(row['native_before_sha256'] == row['native_after_sha256'] and
                row['observation_before_sha256'] == row['observation_after_sha256'], 'observer changed live state/input')
        self.last_time = now
        return row


def reset_geometry_digest(row):
    excluded = {'native_before_sha256', 'native_after_sha256', 'observation_before_sha256',
                'observation_after_sha256', 'copied_state_sha256', 'shadow_copied_state_sha256'}
    return hashlib.sha256(json.dumps({k: v for k, v in row.items() if k not in excluded},
                                     sort_keys=True, allow_nan=False).encode()).hexdigest()
