"""Three scoped CSV projections; unselected rows/weight columns stay lexical."""
from __future__ import annotations
import builtins
import csv
import io
import math
import os
from pathlib import Path
import sys
import zipfile
from experiments import libero_task7_mse_gate_stop_direction_grid_protocol as d
from experiments import libero_task7_optimization_path_decomposition_access as legacy

base,runtime=legacy.base,legacy.runtime
identity=lambda row:{k:row[k] for k in d.IDS}
key=lambda row:tuple(row[k] for k in d.IDS)
CLASSES=['near_zero_total_slope_unresolved','positive_slack_local_feasible_by_continuity','positive_slack_near_numeric_band',
 'boundary_clip_kink_unresolved','boundary_direction_conflict_near_zero_alpha','boundary_descent_local_feasible','boundary_near_zero_mse_slope_unresolved']


def preflight(config=d.DEFAULT_CONFIG,binding=d.DEFAULT_BINDING):return d.validate(config),d.require_binding(config,binding)


def validate_call(cfg,sp,bp):
 frozen,bound=preflight(Path(sp['path']),Path(bp['path']))
 d.p.require(cfg==frozen and d.p.pin(Path(sp['path']))==sp and d.p.pin(Path(bp['path']))==bp and bound['before_research_payload'] is True,'payload API requires unchanged binding')


def typed(name,values):
 d.p.require(list(values)==d.COLUMNS[name],'unlisted projection')
 integers={'task','seed','fold','extra_updates','from_extra_updates','coordinate','trial_index','n',*d.COUNTS}
 booleans={'accepted','armijo_pass','mse_envelope_pass','mse_gate_applied'}
 enums={'arm':d.p.ARMS,'mode':[d.parent.MODE],'condition':['saved500',*[arm+'_final' for arm in d.parent.ARMS]],
  'optimization_arm':d.parent.ARMS,'reason':['accepted','armijo_rejected','mse_envelope_rejected','armijo_and_mse_rejected']}
 row={}
 for field,text in values.items():
  if field in integers:
   value=int(text);d.p.require(str(value)==text,'noncanonical integer');row[field]=value
  elif field in booleans:
   d.p.require(text in ('True','False'),'noncanonical boolean');row[field]=text=='True'
  elif field in enums:d.p.require(text in enums[field],'unlisted enum');row[field]=text
  elif field=='branch_digest':
   d.p.require(len(text)==64 and all(v in '0123456789abcdef' for v in text),'branch digest schema differs');row[field]=text
  else:
   value=float(text);d.p.require(math.isfinite(value),'nonfinite projected scalar');row[field]=value
 d.p.require(row['task']==7 and row['seed'] in d.p.SEEDS and row['fold'] in (0,1,2),'source identity differs')
 return row


class Guard:
 def __init__(self,cfg,outputs=(),limits=None,audit=False):
  self.cfg=cfg;self.refs={k:{**v,'path':str(Path(v['path']).resolve())} for k,v in cfg['inputs'].items()}
  self.paths={r['path'] for r in self.refs.values()};self.outputs={str(Path(p).resolve()) for p in outputs};self.audit=audit
  d.p.require(set(self.refs)==set(d.COLUMNS) and len(self.paths)==3 and not self.paths.intersection(self.outputs)
   and all(Path(p).suffix=='.csv' for p in self.paths|self.outputs),'three disjoint CSV inputs required')
  self.focus={tuple(str(h[k]) for k in d.IDS):str(h['stop_extra_updates']) for h in cfg['heads']}
  self.all_ids=set(self.focus)|{tuple(str(h[k]) for k in d.IDS) for h in cfg['full50_metadata_only']}
  d.p.require(len(self.focus)==11 and len(self.all_ids)==18,'fixed11of18 identity coverage required')
  self.limits=limits or base.Limits(120,268435456);self.telemetry=str(Path('/proc/self/status').resolve())
  self.saved=[];self.denied=[];self.attempted=[];self.completed=[];self.rows={};self.selected_rows={};self.cells={};self.lexical_cells=0;self.hash_checks=[]
  self.active_path=self.hash_path=None;self.reader=csv.reader
 def patch(self,obj,name,value):self.saved.append((obj,name,getattr(obj,name)));setattr(obj,name,value)
 def reject(self,kind,detail=''):
  self.denied.append({'kind':kind,'detail':str(detail)});raise PermissionError('forbidden saved direction/grid access: '+kind+': '+str(detail))
 def hash_one(self,name):
  ref=self.refs[name];self.hash_path=ref['path']
  try:d.p.require(d.p.sha(ref['path'])==ref['sha256'],'source CSV SHA changed')
  finally:self.hash_path=None
  self.hash_checks.append(name)
 def selected(self,name,raw,positions):
  ident=tuple(raw[positions[k]] for k in d.IDS)
  d.p.require(ident in self.all_ids,'unlisted lexical source identity')
  if name in ('endpoints','parameters'):
   condition=raw[positions['condition']]
   d.p.require(condition in ['saved500',*[arm+'_final' for arm in d.parent.ARMS]],'unlisted lexical endpoint role')
   return ident in self.focus and condition in self.cfg['design']['points']
  opt=raw[positions['optimization_arm']]
  d.p.require(opt in d.parent.ARMS,'unlisted lexical optimization arm')
  return ident in self.focus and opt==d.parent.ARMS[1] and raw[positions['from_extra_updates']]==self.focus[ident]
 def project(self,name,columns=None):
  if name not in self.refs:return self.reject('unlisted_CSV',name)
  if columns is not None and list(columns)!=d.COLUMNS[name]:return self.reject('projection_columns',columns)
  if name in self.attempted:return self.reject('repeated_CSV_pass',name)
  self.attempted.append(name);self.rows[name]=self.selected_rows[name]=self.cells[name]=0;self.hash_one(name);ref=self.refs[name];self.active_path=ref['path']
  try:handle=Path(ref['path']).open(newline='')
  finally:self.active_path=None
  with handle:
   reader=self.reader(handle);header=next(reader,None)
   d.p.require(header==ref['columns'] and len(header)==len(set(header)),'source header differs')
   all_positions={k:i for i,k in enumerate(header)};positions={k:all_positions[k] for k in d.COLUMNS[name]}
   for raw in reader:
    d.p.require(len(raw)==len(header),'source width differs');self.rows[name]+=1;self.lexical_cells+=len(raw)
    d.p.require(self.rows[name]<=ref['rows'],'source row budget exceeded')
    if self.rows[name]%256==0:self.limits.check()
    if not self.selected(name,raw,all_positions):continue
    self.selected_rows[name]+=1;self.cells[name]+=len(positions)
    d.p.require(self.selected_rows[name]<=self.cfg['budget']['selected_rows_per_stage'][name],'selected row budget exceeded')
    yield typed(name,{k:raw[i] for k,i in positions.items()})
  d.p.require(self.rows[name]==ref['rows'],'source rows differ');self.completed.append(name)
 def __enter__(self):
  open0,io0,path0,import0=builtins.open,io.open,Path.open,builtins.__import__;file0=io.FileIO;reader0=csv.reader
  def check(path,mode):
   if not isinstance(path,(str,bytes,os.PathLike)):return self.reject('nonpath_IO')
   name=str(Path(os.fsdecode(path)).resolve());write=any(c in mode for c in 'wax+')
   if name in self.paths:
    if write:return self.reject('source_write')
    if name not in (self.active_path,self.hash_path):return self.reject('raw_CSV_bypass')
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
   if isinstance(name,(str,Path)) and str(Path(name).resolve()) in self.paths:return self.reject('direct_CSV_reader')
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
  b=self.cfg['budget'];d.p.require(self.attempted==self.completed==list(d.COLUMNS) and self.rows==b['source_rows_per_stage']
   and self.selected_rows==b['selected_rows_per_stage'] and self.cells==b['projected_cells_per_stage']
   and self.lexical_cells==b['lexical_cells_per_stage'] and not self.denied,'projection budget differs')
  for name in self.refs:self.hash_one(name)
  d.p.require(len(self.hash_checks)==6,'source SHA budget differs');self.limits.check()
 def record(self):
  return dict(source_CSV_passes={'attempted':len(self.attempted),'completed':len(self.completed)},rows=self.rows,selected_rows=self.selected_rows,
   cells=self.cells,lexical_cells=self.lexical_cells,byte_SHA_checks=self.hash_checks,projected_columns=d.COLUMNS,denied_operations=self.denied,
   unselected_rows_or_columns_numeric_converted=0,parameter_weight_column_numeric_converted=0,
   archive_head_checkpoint_reads_including_SHA=0,new_training_forward_gradient_environment_or_sealed_calls=0)
