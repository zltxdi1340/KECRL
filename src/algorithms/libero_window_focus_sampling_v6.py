"""Bounded, local-RNG categorical draws from source window metadata only."""
from __future__ import annotations

from bisect import bisect_right
from dataclasses import dataclass
from itertools import accumulate

ARMS=('uniform_shared_3a','late_half_x3_shared_3a')
SEEDS=(20261106,20261107,20261108)


@dataclass(frozen=True)
class WindowTable:
    ids: tuple
    lengths: tuple
    late: tuple
    weights: tuple
    cumulative: tuple
    quartiles: tuple

    @property
    def total_weight(self):return self.cumulative[-1]

    def index_for_ticket(self,ticket):
        if type(ticket) is not int or not 0<=ticket<self.total_weight:
            raise ValueError('bounded integer ticket required')
        return bisect_right(self.cumulative,ticket)

    def manifest(self):
        return dict(weights=list(self.weights),total_weight=self.total_weight,
            probabilities=[w/self.total_weight for w in self.weights],replacement=True,
            method='integer_ticket_right_bisect',rng_seed_offset=70777)

    def progress(self,indices):
        if len(indices)!=8 or any(type(i) is not int or not 0<=i<len(self.ids) for i in indices):
            raise ValueError('eight bounded window indices required')
        q=[sum(self.quartiles[i][j] for i in indices) for j in range(4)]
        late=sum(self.late[i] for i in indices)
        return dict(early_window_draws=8-late,late_window_draws=late,valid_frames=sum(q),
            early_valid_frames=q[0]+q[1],late_valid_frames=q[2]+q[3],
            valid_frame_quartiles=q,distinct_windows_in_batch=len(set(indices)))


def make_table(window_ids,valid_lengths,demo_lengths,*,arm):
    if arm not in ARMS or not 1<=len(window_ids)<=550 or len(valid_lengths)!=len(window_ids):
        raise ValueError('bounded metadata and fixed sampling arm required')
    ids=[];late=[];weights=[];quartiles=[]
    for identity,L in zip(window_ids,valid_lengths):
        if not isinstance(identity,(tuple,list)) or len(identity)!=2 or any(type(v) is not int for v in identity):
            raise ValueError('integer demo/start metadata required')
        d,s=identity
        if not 0<=d<30 or str(d) not in demo_lengths:
            raise ValueError('source demos0..29 only')
        n=demo_lengths[str(d)]
        if type(n) is not int or n<2 or not 1<=s<n or (s-1)%10 or type(L) is not int or L!=min(10,n-s):
            raise ValueError('causal window/demo endpoint differs')
        is_late=2*s>=n;counts=[0]*4
        for t in range(s,s+L):counts[min(3,4*t//n)]+=1
        ids.append((d,s));late.append(is_late);quartiles.append(tuple(counts))
        weights.append(3 if arm==ARMS[1] and is_late else 1)
    if len(set(ids))!=len(ids):raise ValueError('duplicate source windows')
    return WindowTable(tuple(ids),tuple(valid_lengths),tuple(late),tuple(weights),tuple(accumulate(weights)),tuple(quartiles))


class WindowFocusSampler:
    """Exactly500 eight-window draws; no filesystem, model or global RNG work."""
    def __init__(self,window_ids,valid_lengths,demo_lengths,*,seed,arm):
        import torch
        if seed not in SEEDS:raise ValueError('frozen training seed required')
        self.table=make_table(window_ids,valid_lengths,demo_lengths,arm=arm)
        self.generator=torch.Generator().manual_seed(seed+70777)
        self.completed=0

    def draw(self):
        import torch
        if self.completed>=500:raise RuntimeError('fixed500 sampling budget exhausted')
        tickets=torch.randint(self.table.total_weight,(8,),generator=self.generator).tolist()
        indices=[self.table.index_for_ticket(t) for t in tickets]
        self.completed+=1
        return indices,tickets
