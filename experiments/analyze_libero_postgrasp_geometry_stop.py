"""Bounded postmortem of the rejected observation; no simulation or retry."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import zipfile

import numpy as np

from experiments import libero_postgrasp_geometry_protocol as p


def selected_rgb(item, frames, expected_length):
    p.require(frames==[152,153], 'only the admitted preceding/rejected wrist frames')
    p.require(p.pin(item['path'])==item, 'postmortem archive changed')
    with zipfile.ZipFile(item['path']) as archive:
        p.require(len(archive.namelist())==8 and set(archive.namelist())=={k+'.npy' for k in p.previous.MEMBERS}, 'postmortem archive schema differs')
        with archive.open('eye_in_hand_rgb.npy') as member:
            version=np.lib.format.read_magic(member)
            p.require(version==(1,0), 'unexpected image NPY version')
            shape,fortran,dtype=np.lib.format.read_array_header_1_0(member)
            p.require(shape==(expected_length,128,128,3) and fortran is False and dtype==np.dtype('uint8'), 'image shape/order/dtype differs')
            header=member.tell();skip=frames[0]*49152
            while skip:
                chunk=member.read(min(skip,65536));p.require(bool(chunk), 'truncated image member');skip-=len(chunk)
            raw=member.read(2*49152);p.require(len(raw)==2*49152, 'truncated admitted RGB frames')
            consumed=member.tell();p.require(consumed==header+154*49152 and consumed<=154*49152+256, 'postmortem decompression cap exceeded')
    return np.frombuffer(raw,np.uint8).reshape(2,128,128,3).copy(),dict(pixel_values_decoded=98304,member_uncompressed_bytes_consumed=consumed,header_bytes=header)


def quantify(original,replay,frames):
    rows=[]
    for a,b,step in zip(original,replay,frames):
        delta=b.astype(np.int16)-a.astype(np.int16);coords=np.argwhere(delta!=0)
        rows.append(dict(step=step,exact=bool(np.array_equal(a,b)),different_channels=len(coords),
            different_pixels=int(np.any(delta!=0,axis=2).sum()),maximum_channel_error=int(np.abs(delta).max()),
            mean_channel_error=float(np.abs(delta).mean()),signed_channel_error_sum=int(delta.sum()),
            first_differences=[dict(row=int(y),column=int(x),channel=int(c),original=int(a[y,x,c]),replay=int(b[y,x,c]),delta=int(delta[y,x,c])) for y,x,c in coords[:32]]))
    return rows


def journal_comparison(original,retained):
    def key(r):return (r['problem_id'],r['initial_state_index'],r['step'],r['kind']=='warmup')
    wanted={key(r) for r in retained};p.require(len(wanted)==len(retained),'duplicate retained public event')
    refs={key(r):r for r in original if key(r) in wanted}
    p.require(set(refs)==wanted,'original journal lacks retained event')
    differences=[]
    for row in retained:
        ref=refs[key(row)]
        p.require(row['outcome']==ref['outcome'],'postmortem native outcome mismatch')
        components=[k for k,v in row['component_sha256'].items() if ref['component_sha256'][k]!=v]
        exact=not components
        p.require((row['observation_sha256']==ref['observation_sha256'])==exact
            and row['exact_public']==exact,'component/overall stop gate inconsistent')
        if components:differences.append(dict(problem_id=row['problem_id'],step=row['step'],components=components))
    return dict(public_events=len(retained),matching_public_events=len(retained)-len(differences),differences=differences,
        provenance='journal hash comparison; not a new independent decoding of every image')


def main(config,binding):
    cfg=json.loads(Path(config).read_text());b=json.loads(Path(binding).read_text())
    p.require(cfg['kind']=='libero_postgrasp_geometry_stop_readonly' and cfg['frames']==[152,153]
        and cfg['native_steps']==0 and cfg['image_member']=='eye_in_hand_rgb.npy', 'postmortem scope differs')
    p.require(b['science']==p.pin(config) and b['status']=='bound_before_stop_pixel_access', 'postmortem binding required')
    for item in b['dependencies']+cfg['parents']:
        p.require(p.pin(item['path'])==item, 'postmortem dependency changed')
    base=Path(cfg['output']);p.require(not base.exists() and base.resolve().is_relative_to(p.ROOT/'results'), 'new persistent postmortem output required')
    base.mkdir(parents=True)
    try:
        x,c1=selected_rgb(cfg['original'],cfg['frames'],301);y,c2=selected_rgb(cfg['replay'],cfg['frames'],154)
        rows=quantify(x,y,cfg['frames'])
        p.require(rows[0]['exact'] and not rows[1]['exact'], 'postmortem does not confirm registered first mismatch')
        failure=json.loads(Path(cfg['failure']['path']).read_text())
        oldjournal=[json.loads(line) for line in Path(cfg['original_journal']['path']).read_text().splitlines()]
        newjournal=[json.loads(line) for line in Path(cfg['replay_journal']['path']).read_text().splitlines()]
        journal=journal_comparison(oldjournal,newjournal)
        p.require(journal['public_events']==1987 and journal['matching_public_events']==1986
            and journal['differences']==[dict(problem_id='flip01-confirm5-frame_mlp',step=153,components=['eye_in_hand_rgb'])], 'registered stop differs')
        counts=failure['ledger']['counts']
        p.require(len(failure['records'])==6 and failure['active']['step']==153
            and counts['native_steps']==dict(attempted=2123,completed=2123)
            and counts['saved_actions']==dict(attempted=1953,completed=1953)
            and counts['zero_steps']==dict(attempted=170,completed=170)
            and counts['geometry_samples']==dict(attempted=1209,completed=1209)
            and counts['replay_episodes']==dict(attempted=7,completed=6), 'stop budget manifest differs')
        for record in failure['records']:
            for k in ['public','geometry']:p.require(p.pin(record[k]['path'])==record[k],'completed retained archive changed')
        p.require(p.pin(failure['partial_geometry']['path'])==failure['partial_geometry'],'partial geometry changed')
        result=dict(status='complete_readonly_rejected_wrist_pixel_comparison',science=p.pin(config),binding=p.pin(binding),
            original=cfg['original'],replay=cfg['replay'],frames=rows,actual=dict(source_archives=2,image_members=2,
                pixel_values_decoded=c1['pixel_values_decoded']+c2['pixel_values_decoded'],
                member_uncompressed_bytes_consumed=c1['member_uncompressed_bytes_consumed']+c2['member_uncompressed_bytes_consumed'],
                native_steps=0,policy_forwards=0,initial_bank_reads=0,model_reads=0),
            stop_journal=journal,partial_geometry_independently_audited=False,
            caps_applied_to_separate_postmortem_only=True,completed_geometry_experiment=False,
            root_cause='unresolved: RGB difference does not identify EGL, hidden state or numerical cause',
            replay_gate_relaxed=False,retried=False)
        p.write_new(base/'pixels.json',result);return result
    except BaseException as error:
        p.write_new(base/'failure.json',dict(status='technical_readonly_postmortem_failure',error=repr(error)));raise


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--config',type=Path,required=True);parser.add_argument('--binding',type=Path,required=True)
    args=parser.parse_args();result=main(args.config,args.binding);print(json.dumps(dict(status=result['status'],frames=result['frames'],actual=result['actual'])),flush=True)
