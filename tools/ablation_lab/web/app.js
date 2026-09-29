import {SceneView, clearGeometryCache} from './viewer.js';
const $=s=>document.querySelector(s), $$=s=>[...document.querySelectorAll(s)];
const base=()=>({lens:'anycalib',depth:'moge',camera:'slam',mesh:'provided',pose:'foundationpose',hand:'hamer',crop:'mask',stage:'filled',gravity:'on',refine:'off'});
let configs=[base(),{...base(),refine:'on'}], edit=1, episode, catalog, faces, frame=0, playing=false, lastTick=0, viewModes=['world','world'], loading=0;
let sceneViews=[], sceneAvailable=[false,false], loadedHands=[null,null], loadedVariants=[null,null];
let cameraSync=false;
const stageNames={raw:'Before depth alignment',aligned:'Depth-aligned observations',filled:'Depth-aligned + gap-filled',cleaned:'Custom outlier cleanup',loader_scale:'Loader: scale discarded'};
const definitions=[
 ['lens','Lens correction',[['anycalib','AnyCalib'],['factory','Factory calibration · not run'],['off','Off · not run']]],
 ['depth','Depth + intrinsics',[['moge','MoGe'],['stereo','FoundationStereo · not run']]],
 ['camera','Camera trajectory',[['slam','DROID-SLAM'],['static','Static camera export']]],
 [null,'Initial hand detection','WiLoR'],[null,'Object detection','Grounding DINO'],[null,'Temporal masks','SAM2'],
 ['mesh','Object geometry',[['provided','Provided metric mesh'],['sam3d','SAM3D · not run']]],
 ['pose','Object pose',[['foundationpose','FoundationPose + smoothing']]],
 ['hand','Hand reconstruction',[['hamer','HaMeR'],['hawor','HaWoR · not run']]],
 ['crop','HaMeR crop',[['mask','SAM2 mask bbox'],['detector','WiLoR detector bbox · not run']]],
 ['stage','Geometric alignment',Object.entries(stageNames)],
 ['gravity','World orientation',[['on','GeoCalib · gravity aligned'],['off','No gravity alignment']]],
 ['refine','Joint refinement',[['off','Off'],['on','Gaussian splatting']]],
];
function toast(message){$('#toast').textContent=message;$('#toast').classList.add('show');setTimeout(()=>$('#toast').classList.remove('show'),2500);}
function text(tag,value,cls){const e=document.createElement(tag);e.textContent=value;if(cls)e.className=cls;return e;}
function configurationTitle(c){return c.hand==='hawor'?'HaWoR':c.refine==='on'?'HaMeR + gsplat':c.crop==='detector'?'HaMeR · detector crop':stageNames[c.stage];}
function experiment(c){return Object.values(episode?.experiments||{}).find(e=>Object.entries(e.config).every(([k,v])=>c[k]===v));}
function keyFor(c){return experiment(c)?.variant_key||`${c.refine==='on'?'refined':'baseline'}_${c.camera==='static'?'static':c.gravity==='on'?'gravity':'slam'}`;}
function handFor(c){return experiment(c)?.hand_key||(c.refine==='on'?'refined':c.stage);}
function missing(c){
 if(!episode)return 'Loading episode';
 if(experiment(c))return null;
 if(c.lens!=='anycalib')return 'This lens-correction variant has not been computed. Downstream depth, tracking and reconstruction must be rerun in a fresh output directory.';
 if(c.depth!=='moge')return 'Stereo depth has not been integrated into this ego run. The supplied stereo videos and calibration are available for a separate experiment.';
 if(c.mesh!=='provided')return 'These runs used the supplied metric mesh. A SAM3D reconstruction is a separate run, not an interchangeable display layer.';
 if(c.hand!=='hamer')return 'HaWoR is supported by the pipeline entry point, but no HaWoR output, weights or container were present in these runs. Compute the variant before comparing it.';
 if(c.crop!=='mask')return 'The detector-crop HaMeR variant has not been computed for this episode.';
 if(c.camera==='static'&&c.gravity==='on')return 'No static-camera + gravity-aligned bundle was recorded. Turn gravity alignment off to inspect the saved static-camera export.';
 if(!episode.variants[keyFor(c)])return 'This scene bundle is unavailable. Check the episode status and logs; incomplete runs are not substitutes for completed results.';
 if(!episode.hands[handFor(c)])return 'This hand stage is unavailable for the selected episode. Existing predictions from another stage are not substituted.';
 return null;
}
function pipeline(){
 const root=$('#pipeline');root.replaceChildren();const c=configs[edit];
 definitions.forEach(([key,label,options],i)=>{
  const row=text('div','',`pipelineStage ${key?'':'fixed'}`);row.dataset.number=String(i+1).padStart(2,'0');
  row.append(text('label',label));
  if(key){
   const select=document.createElement('select');select.setAttribute('aria-label',label);select.dataset.key=key;
   for(const [v,l]of options){const computed=Object.values(episode?.experiments||{}).some(e=>e.config[key]===v);const o=text('option',computed?l.replace(' · not run',''):l);o.value=v;select.append(o);}select.value=c[key];
   select.onchange=()=>{c[key]=select.value;if(key==='hand'&&c.hand==='hawor'){const saved=Object.values(episode?.experiments||{}).find(e=>e.config.hand==='hawor');if(saved)Object.assign(c,saved.config);else{c.crop='mask';c.stage='filled';c.refine='off';}}if(key==='stage')c.refine='off';if(key==='refine'&&c.refine==='on')c.stage='filled';if(key==='camera'&&c.camera==='static')c.gravity='off';pipeline();refreshPane(edit);saveView();};row.append(select);
   if(c[key]!==configs[1-edit][key])row.classList.add('changed');
   if((key==='hand'&&c.hand==='hawor'&&!experiment(c))||(key==='crop'&&c.crop==='detector'&&!experiment(c))||(key==='mesh'&&c.mesh==='sam3d')||(key==='depth'&&c.depth==='stereo')||(key==='lens'&&c.lens!=='anycalib'))row.classList.add('unavailable');
  }else{row.append(text('strong',options),text('span','✓','checkmark'));}
  root.append(row);
 });
 $$('[data-edit]').forEach(b=>b.classList.toggle('active',+b.dataset.edit===edit));
}
function metric(value,label,unit='',bad=false){
 const e=text('div','','metric');e.append(text('small',label));const strong=text('strong',value??'—',bad?'bad':'');strong.append(text('em',unit));e.append(strong);return e;
}
function showEmpty(pane,message){const empty=pane.querySelector('.empty');empty.replaceChildren(text('span','NOT COMPUTED','badge warning'),text('strong','No result for this combination'),text('p',message));empty.hidden=false;}
async function refreshPane(index){
 const pane=$(`#pane${index}`),c=configs[index],reason=missing(c);const generation=++sceneViews[index].generation;
 sceneAvailable[index]=false;loadedHands[index]=null;loadedVariants[index]=null;
 pane.querySelector('.paneTitle').textContent=configurationTitle(c);
 pane.querySelector('.paneSubtitle').textContent=`${c.camera==='static'?'Static camera':'DROID-SLAM'} · ${c.gravity==='on'?'gravity aligned':'original axes'} · ${c.mesh==='provided'?'supplied mesh':'SAM3D mesh'}`;
 pane.querySelector('.status').textContent=reason?'NOT COMPUTED':episode.superseded?'ARCHIVED':'SAVED OUTPUT';pane.querySelector('.status').classList.toggle('warning',!!reason||episode.superseded);
 const metrics=pane.querySelector('.metrics');metrics.replaceChildren();
 if(reason){showEmpty(pane,reason);pane.querySelector('.stageVideo').hidden=true;pane.querySelector('.stageVideo').pause();provenance();drawChart();return;}
 pane.querySelector('.empty').hidden=false;pane.querySelector('.empty').replaceChildren(text('p','Loading per-frame geometry…'));
 const hand=episode.hands[handFor(c)],variant=episode.variants[keyFor(c)];
 loadedHands[index]=hand;loadedVariants[index]=variant;
 const rmse=variant.metrics.object_rigid_rmse_mm;
 const max=Math.max(hand.left.max_camera_distance_m||0,hand.right.max_camera_distance_m||0);
 const coverage=(hand.left.coverage+hand.right.coverage)*50;
 metrics.append(metric(rmse?.toFixed(1),'OBJECT PATH · RIGID RMSE','mm'),metric(max.toFixed(2),'MAX WRIST–CAMERA DISTANCE','m',max>1.1),metric(coverage.toFixed(0),'HAND FRAME COVERAGE','%'));
 try{
   await sceneViews[index].load(episode,hand,variant,faces);
   // load() increments once; a newer selection must not reveal stale geometry.
   if(sceneViews[index].generation!==generation+1)return;
   sceneAvailable[index]=true;pane.querySelector('.empty').hidden=true;sceneViews[index].frame(frame);sceneViews[index].setTruth($('#gt').checked);
   setView(index,viewModes[index]);
 }catch(error){showEmpty(pane,`Geometry failed to load: ${error.message}`);console.error(error);}
 provenance();drawChart();
}
function setView(index,mode){
 viewModes[index]=mode;const pane=$(`#pane${index}`);const reason=missing(configs[index]);
 pane.querySelectorAll('[data-view]').forEach(b=>b.classList.toggle('active',b.dataset.view===mode));
 const video=pane.querySelector('.stageVideo');video.pause();const image=pane.querySelector('.stageImage');image.hidden=true;
 if(reason){showEmpty(pane,reason);return;}
 let url=null,caption='';
 if(mode==='ego'){url=episode.stages[handFor(configs[index])];caption='RECORDED EGO OVERLAY';}
 if(mode==='lens'){url=experiment(configs[index])?null:episode.undistorted;caption='ANYCALIB UNDISTORTED INPUT · SHARED UPSTREAM STAGE';}
 if(mode==='depth'){caption='MOGE DEPTH · FIXED SCALE: 0.1 m BLUE → 2.0 m RED (CLIPPED)';}
 if(mode==='masks'){url=experiment(configs[index])?null:episode.stages.masks;caption='SHARED SAM2 MASKS · BEFORE HAND ABLATIONS';}
 if(mode==='retarget'){url=episode.retarget?.video;caption='EXISTING BASELINE IK · INDEPENDENT OF A/B';}
 pane.querySelector('.scene').hidden=mode!=='world';video.hidden=mode==='world';
 pane.querySelector('.legend').hidden=mode!=='world';
 pane.querySelector('.viewLabel').replaceChildren(text('span',mode==='world'?'WORLD SPACE':caption),text('span',mode==='world'?'Drag to orbit · scroll to zoom':''));
 if(mode==='depth'){
  video.hidden=true;
  if(!episode.depth_preview||experiment(configs[index])){showEmpty(pane,'No matching depth preview was exported for this run.');return;}
  image.hidden=false;image.src=episode.depth_preview+String(frame).padStart(6,'0')+'.jpg';
 }else if(mode!=='world'){
  if(!url){showEmpty(pane,'No recorded video exists for this stage. The 3D scene still displays the saved per-frame geometry.');return;}
  if(video.getAttribute('src')!==url){video.src=url;video.load();}
  video.onloadedmetadata=()=>syncVideo(video);video.onseeked=()=>{if(!playing)syncVideo(video);};syncVideo(video);
 }
 pane.querySelector('.empty').hidden=true;
 if(mode==='retarget'&&episode.retarget){toast('IK replay is the existing baseline run, not regenerated by A/B controls.');}
}
function syncVideo(video){
 if(!episode||!Number.isFinite(video.duration)||video.duration<=0||video.seeking)return;
 const timing=episode.video_timing?.[video.getAttribute('src')];
 const target=timing?.fps?frame/timing.fps:frame/episode.frames*video.duration;
 const t=Math.min(target,Math.max(0,video.duration-.025));
 if(Math.abs(video.currentTime-t)>.015)video.currentTime=t;
}
function setFrame(value){
 if(!episode)return;frame=Math.max(0,Math.min(episode.frames-1,Math.round(value)));$('#timeline').value=frame;
 const seconds=frame/episode.fps;$('#time').textContent=`${Math.floor(seconds/60).toString().padStart(2,'0')}:${(seconds%60).toFixed(2).padStart(5,'0')}`;
 $('#frameLabel').textContent=`${frame} / ${episode.frames-1}`;
 sceneViews.forEach((v,i)=>{if(sceneAvailable[i])v.frame(frame);});
 syncVideo($('#source'));$$('.stageVideo').filter(v=>!v.hidden).forEach(syncVideo);drawChart();
 sceneViews.forEach((_,i)=>{const image=$(`#pane${i} .stageImage`);if(!image.hidden&&episode.depth_preview)image.src=episode.depth_preview+String(frame).padStart(6,'0')+'.jpg';});
}
function drawChart(){
 const canvas=$('#chart'),rect=canvas.getBoundingClientRect(),ratio=Math.min(devicePixelRatio,2);canvas.width=rect.width*ratio;canvas.height=rect.height*ratio;
 const ctx=canvas.getContext('2d');ctx.scale(ratio,ratio);const w=rect.width,h=rect.height,L=40,T=10,R=15,B=24;
 const series=loadedHands.flatMap((h,i)=>h?['left','right'].map((side,j)=>({v:h[side].camera_distance_m,color:i?'#81b9ff':'#bcf779',dashed:j===1})):[]);
 const ymax=Math.max(1.3,...series.flatMap(s=>s.v.filter(x=>x!==null)))*1.05;
 const x=i=>L+i/Math.max(1,(episode?.frames||2)-1)*(w-L-R),y=v=>h-B-v/ymax*(h-T-B);
 ctx.font='9px ui-monospace,monospace';ctx.fillStyle='#718594';ctx.strokeStyle='#25323d';ctx.lineWidth=1;
 for(let i=0;i<4;i++){const v=ymax*i/3;ctx.beginPath();ctx.moveTo(L,y(v));ctx.lineTo(w-R,y(v));ctx.stroke();ctx.fillText(v.toFixed(1)+'m',4,y(v)+3);}
 ctx.fillText('0',L,h-6);ctx.fillText(`${episode?.frames-1||0} frames`,w-76,h-6);
 ctx.strokeStyle='#8b6847';ctx.setLineDash([3,4]);ctx.beginPath();ctx.moveTo(L,y(1.1));ctx.lineTo(w-R,y(1.1));ctx.stroke();ctx.setLineDash([]);
 for(const s of series){ctx.strokeStyle=s.color;ctx.lineWidth=1.4;ctx.setLineDash(s.dashed?[4,3]:[]);ctx.beginPath();let pen=false;s.v.forEach((v,i)=>{if(v===null){pen=false;return;}if(!pen)ctx.moveTo(x(i),y(v));else ctx.lineTo(x(i),y(v));pen=true;});ctx.stroke();}
 ctx.setLineDash([]);ctx.strokeStyle='#e9f0f7';ctx.lineWidth=1;ctx.beginPath();ctx.moveTo(x(frame),T);ctx.lineTo(x(frame),h-B);ctx.stroke();
}
function command(c){
 const id=episode.id;const suffix=Object.entries(c).map(([k,v])=>`${k}-${v}`).join('_');
 if(c.lens==='factory'||c.depth==='stereo')return 'Requires an explicit calibrated/stereo input adapter. This combination is not supported by the current ego CLI; do not relabel an existing MoGe run.';
 const args=['cd /data1/SRL_V2D/reconstruction','/home/exaFLOPs/miniforge3/envs/v2d/bin/python modules/v2d_pipelines/run_ego_reconstruction.py',
  `  --video /data1/video_to_data_challenge/track_3/public/videos/chunk-000/observation.images.ego_cam_a/${id}.mp4`,
  `  --output_dir /data1/SRL_V2D/.ablation_lab/runs/${id}/${suffix}`,`  --object_prompt '${episode.prompt.replaceAll("'","'\\''")}'`,
  `  --hand_tracking ${c.hand} --reference_frame ${episode.reference_frame}`];
 if(c.hand==='hamer')args.push(`  --hamer_crop_source ${c.crop}`);
 if(c.mesh==='provided')args.push(`  --object_mesh /data1/video_to_data_challenge/runs/prep/meshes/${episode.object}/${episode.object}.obj --skip_object_scale_estimation`);
 if(c.lens==='anycalib')args.push('  --undistort');if(c.camera==='slam')args.push('  --run_droid_slam');if(c.gravity==='on')args.push('  --run_gravity_alignment');if(c.refine==='on')args.push('  --run_gsplat_refinement');args.push('  --export_threejs_result');
 return args[0]+'\n'+args.slice(1).join(' \\\n');
}
function provenance(){
 if(!episode)return;const root=$('#provenance');root.replaceChildren();
 configs.forEach((c,i)=>{root.append(text('h3',`${i?'B':'A'} · ${configurationTitle(c)}`));const v=episode.variants[keyFor(c)];root.append(text('p',missing(c)||`Source: ${v.bundle}`));
  if(!missing(c))root.append(text('p',v.checks.length?'Checks: '+v.checks.join('; '):'Frame count, finite values, and camera/world transform consistency passed.'));
  const rt=experiment(c)?.provenance?.retarget;
  if(rt)root.append(text('p',`Fresh kinematic IK: left mean ${rt.left_mean_mm.toFixed(2)} mm (p95 ${rt.left_p95_frame_mean_mm.toFixed(2)}), right mean ${rt.right_mean_mm.toFixed(2)} mm (p95 ${rt.right_p95_frame_mean_mm.toFixed(2)}). Robot QA video has not been rendered.`));
  const pre=text('pre',command(c));root.append(pre);
 });
 if(episode.retarget){root.append(text('h3','Existing kinematic retargeting'));root.append(text('p',episode.retarget.provenance));root.append(text('p',`Stored IK residual (mean): left ${episode.retarget.left.ik_mean_mm.toFixed(1)} mm, right ${episode.retarget.right.ik_mean_mm.toFixed(1)} mm. FPS ${episode.retarget.fps.toFixed(4)}.`));root.append(text('pre',episode.retarget.source));}
 if(episode.log){const a=text('a','Open original run log ↗');a.href=episode.log;a.target='_blank';root.append(a);}
}
function saveView(){if(!episode)return;const q=new URLSearchParams({episode:episode.index,frame:String(frame),a:JSON.stringify(configs[0]),b:JSON.stringify(configs[1])});history.replaceState(null,'','?'+q);}
async function selectEpisode(id,initialFrame=0){
 const token=++loading;playing=false;$('#play').textContent='▶';sceneAvailable=[false,false];loadedHands=[null,null];clearGeometryCache();
 const response=await fetch(`/data/${id}/episode.json`);if(!response.ok)throw new Error('Episode file unavailable');const e=await response.json();if(token!==loading)return;episode=e;
 $('#episode').value=id;$('#timeline').max=e.frames-1;$('#source').src=e.source;$('#source').onloadedmetadata=()=>syncVideo($('#source'));$('#source').onseeked=()=>{if(!playing)syncVideo($('#source'));};
 $('#sourceCaption').textContent=`Original input · ${e.width} × ${e.height}`;$('#episodeMeta').textContent=`${e.frames} frames · ${e.fps.toFixed(2)} fps · ${(e.frames/e.fps).toFixed(1)}s`;
 $('#episodeNotice').hidden=e.status==='complete';$('#episodeNotice').textContent=e.superseded?'Archived baseline: the newer reference-frame rerun did not complete. Baseline views show the earlier run.':'Incomplete reconstruction baseline. Source video and logs are available. Only separately completed, registered experiments can supply new 3D results.';
 $('#findings').replaceChildren(...e.findings.map(f=>text('li',f)));
 const probeButton=$('#cropProbeButton');probeButton.hidden=!e.crop_probe;
 if(e.crop_probe){probeButton.onclick=()=>{
  setFrame(e.crop_probe.frame);const body=$('#cropProbeBody');body.replaceChildren();
  for(const [name,r]of Object.entries(e.crop_probe.results)){
   const card=text('article','');card.append(text('h3',name==='baseline_sam2_crop'?'SAM2-driven crop':'WiLoR detector crop'));
   const img=document.createElement('img');img.src=r.image;img.alt=name+' at frame 244, crop boundary and unoccluded predicted mesh projection';card.append(img);
   card.append(text('strong',r.camera_distance_m.toFixed(2)+' m from camera'));body.append(card);
  }$('#probe').showModal();
 };}
 if(e.retarget)$('#findings').append(text('li',e.retarget.provenance+' Changing A/B settings does not regenerate robot motion.'));
 frame=Math.min(initialFrame,e.frames-1);pipeline();await Promise.all([refreshPane(0),refreshPane(1)]);if(token!==loading)return;setFrame(frame);saveView();
}
function preset(name){
 configs=[base(),base()];
 if(name==='crop')configs[1].crop='detector';
 if(name==='refine')configs[1].refine='on';
 if(name==='depth'){configs[0].stage='raw';configs[1].stage='aligned';}
 if(name==='fill')configs[0].stage='aligned';
 if(name==='clean')configs[1].stage='cleaned';
 if(name==='scale'){configs[0].stage=episode.hands.cleaned?'cleaned':'filled';configs[1].stage='loader_scale';}
 if(name==='slam'){configs[0].camera='static';configs[0].gravity='off';configs[1].gravity='off';}
 $$('[data-preset]').forEach(b=>b.classList.toggle('active',b.dataset.preset===name));pipeline();Promise.all([refreshPane(0),refreshPane(1)]).then(()=>setFrame(frame));saveView();
}
async function start(){
 [catalog,faces]=await Promise.all([fetch('/data/catalog.json').then(r=>r.json()),fetch('/data/hand_faces.json').then(r=>r.json())]);
 const complete=catalog.episodes.filter(e=>e.status==='complete').length;$('#datasetState').textContent=`${catalog.episodes.length} episodes · ${complete} complete`;
 for(const e of catalog.episodes){const o=text('option',`EP ${String(e.index).padStart(2,'0')} · ${e.task}${e.status==='complete'?'':` [${e.status}]`}`);o.value=e.id;$('#episode').append(o);}
 const info=$('#methodText');Object.entries(catalog.methodology).forEach(([k,v])=>info.append(text('p',v)));
 for(let i=0;i<2;i++){
  const pane=$(`#pane${i}`),view=new SceneView(pane.querySelector('.scene'));sceneViews.push(view);
  const stageImage=document.createElement('img');stageImage.className='stageImage';stageImage.alt='MoGe per-frame metric depth, fixed 0.1 to 2 metre colour scale';stageImage.hidden=true;pane.querySelector('.viewport').append(stageImage);
  for(const [mode,label]of [['lens','Lens'],['depth','Depth']]){const b=text('button',label);b.dataset.view=mode;pane.querySelector('.viewControls').insertBefore(b,pane.querySelector('.reset'));}
  view.onCameraChange=()=>{if(cameraSync||!$('#linkCameras').checked||!sceneViews[1-i]||!sceneAvailable.every(Boolean))return;cameraSync=true;sceneViews[1-i].copyCamera(view);cameraSync=false;};
  pane.querySelectorAll('[data-view]').forEach(b=>b.onclick=()=>setView(i,b.dataset.view));pane.querySelector('.reset').title='Fit all geometry in current frame, including outliers';pane.querySelector('.reset').onclick=()=>view.fitFrame(frame);
 }
 const params=new URLSearchParams(location.search);
 try{for(const[i,k]of ['a','b'].entries())if(params.has(k)){const c=JSON.parse(params.get(k));for(const[key,,opts]of definitions)if(key&&opts.some(([v])=>v===c[key]))configs[i][key]=c[key];}}catch{}
 pipeline();$$('[data-edit]').forEach(b=>b.onclick=()=>{edit=+b.dataset.edit;pipeline();});
 $('#episode').onchange=()=>selectEpisode($('#episode').value).catch(fatal);$$('[data-preset]').forEach(b=>b.onclick=()=>preset(b.dataset.preset));
 $('#swap').onclick=()=>{configs.reverse();pipeline();refreshPane(0);refreshPane(1);saveView();};
 $('#timeline').oninput=()=>setFrame(+$('#timeline').value);$('#timeline').onchange=saveView;
 $('#play').onclick=()=>{playing=!playing;lastTick=performance.now();$('#play').textContent=playing?'Ⅱ':'▶';};$('#prev').onclick=()=>setFrame(frame-1);$('#next').onclick=()=>setFrame(frame+1);
 $('#gt').onchange=()=>sceneViews.forEach(v=>v.setTruth($('#gt').checked));
 $('#worst').onclick=()=>{let max=-1,best=0;for(const h of loadedHands.filter(Boolean))for(const side of ['left','right'])h[side].camera_distance_m.forEach((v,i)=>{if(v!==null&&v>max){max=v;best=i;}});setFrame(best);saveView();};
 $('#chart').onclick=e=>{const r=e.currentTarget.getBoundingClientRect();setFrame((e.clientX-r.left-40)/(r.width-55)*(episode.frames-1));};
 $('#share').onclick=async()=>{saveView();try{await navigator.clipboard.writeText(location.href);toast('Comparison link copied');}catch{toast('Copy the current address from your browser');}};
 $('#methodology').onclick=()=>$('#info').showModal();$('#info .close').onclick=()=>$('#info').close();
 $('#probe .close').onclick=()=>$('#probe').close();
 document.addEventListener('keydown',e=>{if(['INPUT','SELECT','TEXTAREA','BUTTON'].includes(e.target.tagName)||document.querySelector('dialog[open]'))return;if(e.code==='Space'){e.preventDefault();$('#play').click();}if(e.code==='ArrowRight')setFrame(frame+1);if(e.code==='ArrowLeft')setFrame(frame-1);});
 const id=catalog.episodes.find(e=>e.index===Number(params.get('episode')??41))?.id||catalog.episodes[0].id;
 await selectEpisode(id,Number(params.get('frame')||0));
 requestAnimationFrame(animate);
}
function animate(now){
 if(playing&&episode){const interval=1000/(episode.fps*Number($('#speed').value));if(now-lastTick>=interval){const steps=Math.floor((now-lastTick)/interval);setFrame((frame+steps)%episode.frames);lastTick+=steps*interval;}}
 sceneViews.forEach(v=>v.render());requestAnimationFrame(animate);
}
function fatal(error){console.error(error);$('#datasetState').textContent='Load error';$('#episodeNotice').hidden=false;$('#episodeNotice').textContent=error.message;}
window.addEventListener('resize',drawChart);
start().catch(fatal);
