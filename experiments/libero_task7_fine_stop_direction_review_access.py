"""One source-JSON parse, explicit saved-value projection and no array access."""
from __future__ import annotations
import builtins
import csv
import io
import json
import math
import os
from pathlib import Path
import sys
import zipfile
from experiments import libero_task7_fine_stop_direction_review_protocol as d
from experiments import libero_task7_optimization_path_decomposition_access as legacy

base,runtime=legacy.base,legacy.runtime
identity=lambda row:{k:row[k] for k in d.IDS}
key=lambda row:tuple(row[k] for k in d.IDS)
CLASSES=['near_zero_total_slope_unresolved','positive_slack_local_feasible_by_continuity','positive_slack_near_numeric_band',
 'boundary_clip_kink_unresolved','boundary_direction_conflict_near_zero_alpha','boundary_descent_local_feasible','boundary_near_zero_mse_slope_unresolved']


def typed(name,row):
 d.p.require(list(row)==d.COLUMNS[name],'unlisted projected fields')
 integers={'task','seed','fold','extra_updates','from_extra_updates','coordinate','trial_index','n',*d.COUNTS}
 booleans={'accepted','armijo_pass','mse_envelope_pass','mse_gate_applied'}
 enums={'arm':d.p.ARMS,'mode':[d.parent.source.MODE],'condition':['saved500','fine20_final'],
  'optimization_arm':['fine20'],'reason':['accepted','armijo_rejected','mse_envelope_rejected','armijo_and_mse_rejected']}
 for f,v in row.items():
  if f in integers:d.p.require(type(v) is int,'integer projected value required')
  elif f in booleans:d.p.require(type(v) is bool,'boolean projected flag required')
  elif f in enums:d.p.require(v in enums[f],'projected role differs')
  elif f=='branch_digest':d.p.require(type(v) is str and len(v)==64 and all(c in '0123456789abcdef' for c in v),'branch digest schema differs')
  else:d.p.require(type(v) in (int,float) and math.isfinite(v),'finite projected scalar required')
 return row


def preflight(config=d.DEFAULT_CONFIG,binding=d.DEFAULT_BINDING):return d.validate(config),d.require_binding(config,binding)


def validate_call(cfg,sp,bp):
 frozen,bound=preflight(Path(sp['path']),Path(bp['path']))
 d.p.require(cfg==frozen and d.p.pin(Path(sp['path']))==sp and d.p.pin(Path(bp['path']))==bp and bound['before_research_payload'] is True,'unchanged fine-stop binding required')


class Guard:
 def __init__(self,cfg,outputs=(),limits=None,audit=False):
  self.cfg=cfg;self.source=cfg['inputs']['source_JSON'];self.paths={str(Path(self.source['path']).resolve())}
  self.outputs={str(Path(p).resolve()) for p in outputs};self.audit=audit
  d.p.require(not self.paths.intersection(self.outputs) and len(self.paths)==1,'one disjoint parent JSON required')
  self.limits=limits or base.Limits(120,268435456);self.telemetry=str(Path('/proc/self/status').resolve())
  self.saved=[];self.denied=[];self.parses=0;self.hash_checks=0;self.payload_depth=self.hash_depth=0
  self.focus={key(h):h for h in cfg['heads']}
  d.p.require(len(self.focus)==5,'fixed5 identity coverage required')
  self.rows={};self.cells={}
 def patch(self,obj,name,value):self.saved.append((obj,name,getattr(obj,name)));setattr(obj,name,value)
 def reject(self,kind,detail=''):
  self.denied.append(dict(kind=kind,detail=str(detail)));raise PermissionError('forbidden fine-stop review access: '+kind+': '+str(detail))
 def hash_one(self):
  self.hash_depth+=1
  try:d.p.require(d.p.sha(self.source['path'])==self.source['sha256'],'parent JSON bytes changed')
  finally:self.hash_depth-=1
  self.hash_checks+=1
 def project(self,columns=None):
  if columns is not None and columns!=d.COLUMNS:return self.reject('unlisted_projection')
  if self.parses:return self.reject('repeated_parent_JSON_parse')
  self.hash_one();self.payload_depth+=1
  try:raw=json.loads(Path(self.source['path']).read_text())
  finally:self.payload_depth-=1
  self.parses+=1
  d.p.require(raw['status']=='complete_mse_guard_finer_grid_pending_audit' and raw['formal_result'] is False
   and raw['science_config']==d.PARENTS['science'] and raw['binding']==d.PARENTS['binding']
   and raw['original_Policy_CNN_environment_or_sealed_calls']==0,'parent JSON provenance differs')
  d.p.require({k:len(raw[k]) for k in self.cfg['budget']['source_rows_per_stage']}==self.cfg['budget']['source_rows_per_stage'],
   'parent JSON coverage differs')
  allids={(7,s,a,f,d.parent.source.MODE) for s in d.p.SEEDS for a in d.p.ARMS for f in range(3)}
  pairs={key(row):row for row in raw['pairs']}
  d.p.require(len(pairs)==18 and set(pairs)==allids
   and [sum(row['cohort']==c for row in pairs.values()) for c in ('zero','short','full50')]==[6,5,7]
   and {k for k,r in pairs.items() if r['cohort']=='short'}==set(self.focus),'fixed five-of18 metadata differs')
  for k,h in self.focus.items():
   row=pairs[k];d.p.require(row['candidate_updates']==row['fine_updates']==h['stop_extra_updates']
    and row['candidate_stop']=='line_search_exhausted' and row['grid_opened'] is True and row['prefix_valid'] is True,
    'selected fine20 stop/prefix differs')
  data={name:[] for name in d.COLUMNS}
  for row in raw['endpoints']:
   d.p.require(key(row) in allids and row['condition'] in ('saved500','coarse12_final','fine20_final'),'unlisted endpoint identity/role')
   if key(row) not in self.focus or row['condition'] not in self.cfg['design']['points']:continue
   data['endpoints'].append(typed('endpoints',{f:row[f] for f in d.COLUMNS['endpoints']}))
   d.p.require(all(type(row[f]) is list and len(row[f])==257 for f in d.VECTORS),'cached257 gradient coordinates required')
   for j in range(257):
    data['parameters'].append(typed('parameters',{**identity(row),'condition':row['condition'],'coordinate':j,**{f:row[f][j] for f in d.VECTORS}}))
  for row in raw['trials']:
   d.p.require(key(row) in allids and row['optimization_arm'] in ('coarse12','fine20'),'unlisted trial identity/arm')
   if key(row) in self.focus and row['optimization_arm']=='fine20' and row['from_extra_updates']==self.focus[key(row)]['stop_extra_updates']:
    data['trials'].append(typed('trials',{f:row[f] for f in d.COLUMNS['trials']}))
  self.rows={k:len(v) for k,v in data.items()};self.cells={k:len(v)*len(d.COLUMNS[k]) for k,v in data.items()}
  d.p.require(self.rows==self.cfg['budget']['selected_rows_per_stage'] and self.cells==self.cfg['budget']['projected_cells_per_stage'],
   'parent saved-value projection budget differs')
  self.limits.check();return data
 def __enter__(self):
  open0,io0,path0,import0=builtins.open,io.open,Path.open,builtins.__import__;file0=io.FileIO;reader0=csv.reader
  def check(path,mode):
   if not isinstance(path,(str,bytes,os.PathLike)):return self.reject('nonpath_IO')
   name=str(Path(os.fsdecode(path)).resolve());write=any(c in mode for c in 'wax+')
   if name in self.paths:
    if write:return self.reject('source_write')
    if not (self.payload_depth or self.hash_depth):return self.reject('raw_JSON_bypass')
   elif name==self.telemetry and not write:return
   elif name in self.outputs:
    if write and self.audit:return self.reject('audit_output_write')
   else:return self.reject('unlisted_read_write_or_archive_SHA',name)
  def opened(path,mode='r',*args,**kw):check(path,mode);return open0(path,mode,*args,**kw)
  def io_opened(path,mode='r',*args,**kw):check(path,mode);return io0(path,mode,*args,**kw)
  def path_opened(path,mode='r',*args,**kw):check(path,mode);return path0(path,mode,*args,**kw)
  class ScopedFileIO(file0):
   def __init__(self,path,mode='r',*args,**kw):check(path,mode);super().__init__(path,mode,*args,**kw)
  def imported(name,*args,**kw):
   if name.split('.')[0] in {'numpy','torch','h5py','pandas','pickle','mmap','libero','mujoco','robosuite','src'}:return self.reject('runtime_import',name)
   return import0(name,*args,**kw)
  def reader(handle,*args,**kw):
   name=getattr(handle,'name',None)
   if isinstance(name,(str,Path)) and str(Path(name).resolve()) in self.paths:return self.reject('direct_source_reader')
   return reader0(handle,*args,**kw)
  deny=lambda *args,**kw:self.reject('array_model_or_alternate_reader')
  for obj,name,value in [(builtins,'open',opened),(io,'open',io_opened),(Path,'open',path_opened),(io,'FileIO',ScopedFileIO),(os,'open',deny),(builtins,'__import__',imported),(csv,'reader',reader),(zipfile.ZipFile,'open',deny)]:self.patch(obj,name,value)
  if 'numpy' in sys.modules:
   np=sys.modules['numpy']
   for obj,name in [(np,'load'),(np,'fromfile'),(np,'loadtxt'),(np,'genfromtxt'),(np.memmap,'__new__'),(np.lib.npyio.NpzFile,'__getitem__'),(np.lib.format,'read_array')]:self.patch(obj,name,deny)
  if 'torch' in sys.modules:
   torch=sys.modules['torch']
   for obj,name in [(torch,'load'),(torch.autograd,'grad'),(torch.autograd,'backward'),(torch.optim.Optimizer,'__init__'),(torch.nn.Module,'__call__')]:self.patch(obj,name,deny)
  return self
 def __exit__(self,*args):
  for obj,name,value in reversed(self.saved):setattr(obj,name,value)
 def complete(self):
  d.p.require(self.parses==1 and self.rows==self.cfg['budget']['selected_rows_per_stage']
   and self.cells==self.cfg['budget']['projected_cells_per_stage'] and not self.denied,'fine-stop projection ledger differs')
  self.hash_one();d.p.require(self.hash_checks==2,'parent JSON SHA budget differs');self.limits.check()
 def record(self):
  return dict(parent_JSON_parses=self.parses,parent_JSON_SHA_checks=self.hash_checks,
   selected_rows=self.rows,projected_cells=self.cells,cached_gradient_coordinates=10280,
   physically_parsed_whole_parent_JSON=True,unselected_records_and_parameter_fields_excluded_from_numeric_projection=True,
   source_CSV_array_checkpoint_HDF5_RGB_reads_including_SHA=0,new_training_forward_gradient_environment_or_sealed_calls=0,
   denied_operations=self.denied)
