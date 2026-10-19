"""Configure checkpoint-history galleries without changing stored run data."""
from pathlib import Path
import json,uuid,wandb
from wandb_workspaces._graphql import execute_graphql
b=Path(__file__).resolve().parents[1];api=wandb.Api()
query='''query {project(name:"v2d-track3-uniretarget",entityName:"DexDreamer"){allViews(viewType:"project-view"){edges{node{id name displayName spec}}}}}'''
mutation='''mutation UpsertView2($id:ID,$entityName:String,$projectName:String,$type:String,$name:String,$displayName:String,$spec:String){upsertView(input:{id:$id,entityName:$entityName,projectName:$projectName,type:$type,name:$name,displayName:$displayName,spec:$spec,createdUsing:WANDB_SDK}){view{id name} inserted}}'''
keys={'videos/physics_checkpoint':'Physics — checkpoint history','videos/assisted_voc':'Assisted VOC — checkpoint history','videos/reference_overlay':'Reference refinement — version history'}
def configure(config,key):
 config.pop('stepIndex',None)
 config.update(chartTitle=keys[key],mediaKeys=[key],mode='gallery',gallerySettings={'axis':'step'},columnCount=1,stepStrideLength=1,snapToExistingStep=True)
r=execute_graphql(api,query,{})
(b/'workspace_before_video_history.json').write_text(json.dumps(r,indent=2))
changed=[]
for edge in r['project']['allViews']['edges']:
 n=edge['node']
 if n['name'] not in ['nw-nwuserexaflops-w','nw-h6fegw1wdt4-v']:continue
 spec=json.loads(n['spec']);bank=spec['section']['panelBankConfig'];overrides=bank.setdefault('panelConfigOverrides',{})
 for key in keys:configure(overrides.setdefault(key,{}).setdefault('config',{}),key)
 for section in bank.get('sections',[]):
  if section.get('name','').lower()!='videos':continue
  section['isOpen']=True
  section.setdefault('flowConfig',{})['rowsPerPage']=2
  found=set()
  for panel in section.get('panels',[]):
   cfg=panel.get('config',{});media=cfg.get('mediaKeys',[])
   for key in keys:
    if key in media:configure(cfg,key);found.add(key)
  if not section.get('isPanelsAuto',False):
   for key in keys.keys()-found:
    cfg={};configure(cfg,key)
    section.setdefault('panels',[]).append({'__id__':uuid.uuid4().hex[:11],'layout':{'x':0,'y':0,'w':8,'h':8},'viewType':'Media Browser','config':cfg})
 execute_graphql(api,mutation,dict(id=n['id'],entityName='DexDreamer',projectName='v2d-track3-uniretarget',type='project-view',name=n['name'],displayName=n['displayName'],spec=json.dumps(spec)))
 changed.append(n['name'])
r=execute_graphql(api,query,{})
(b/'workspace_after_video_history.json').write_text(json.dumps(r,indent=2))
for edge in r['project']['allViews']['edges']:
 n=edge['node']
 if n['name'] not in changed:continue
 bank=json.loads(n['spec'])['section']['panelBankConfig']
 for key in keys:
  cfg=bank['panelConfigOverrides'][key]['config']
  assert cfg['gallerySettings']['axis']=='step' and cfg['snapToExistingStep'] and 'stepIndex' not in cfg
print(json.dumps({'updated_workspaces':changed,'video_keys':list(keys),'mode':'gallery by checkpoint step, one column','verified_saved_configuration':True}))
