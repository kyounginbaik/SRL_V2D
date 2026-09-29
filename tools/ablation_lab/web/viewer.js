import * as THREE from '/data/vendor/three.module.js';
import {OrbitControls} from '/data/vendor/OrbitControls.js';

const buffers = new Map();
export async function loadBuffer(spec) {
  if (!buffers.has(spec.url)) buffers.set(spec.url, fetch(spec.url).then(r => {
    if (!r.ok) throw new Error(`Missing geometry: ${spec.url}`);
    return r.arrayBuffer();
  }).then(b => spec.dtype === '<u4' ? new Uint32Array(b) : new Float32Array(b)));
  return buffers.get(spec.url);
}
export function clearGeometryCache() { buffers.clear(); }
function matrix(rows) { return new THREE.Matrix4().set(...rows.flat()); }
function setMatrix(object, m) { object.matrixAutoUpdate = false; object.matrix.copy(m); }

export class SceneView {
  constructor(container) {
    this.container = container;
    THREE.Object3D.DEFAULT_UP.set(0, 0, 1);
    this.renderer = new THREE.WebGLRenderer({antialias:true, alpha:false});
    this.renderer.setPixelRatio(Math.min(devicePixelRatio, 1.5));
    this.renderer.setClearColor(0x0e161e);
    container.append(this.renderer.domElement);
    this.scene = new THREE.Scene();
    this.scene.fog = new THREE.Fog(0x0e161e, 5, 18);
    this.camera = new THREE.PerspectiveCamera(42, 1, .005, 100);
    this.camera.up.set(0,0,1);
    this.controls = new OrbitControls(this.camera, this.renderer.domElement);
    this.controls.enableDamping = true;
    this.controls.dampingFactor = .12;
    this.controls.addEventListener('change', () => this.onCameraChange?.());
    this.scene.add(new THREE.HemisphereLight(0xd7e7f8,0x566064,2));
    const key = new THREE.DirectionalLight(0xffffff,2.2); key.position.set(1,-2,4); this.scene.add(key);
    const fill = new THREE.DirectionalLight(0x9ac3f5,.8); fill.position.set(-2,1,1); this.scene.add(fill);
    this.content = new THREE.Group(); this.scene.add(this.content);
    this.grid = new THREE.GridHelper(8,80,0x354651,0x263540); this.grid.rotation.x = Math.PI/2; this.scene.add(this.grid);
    this.grid.material.transparent=true;this.grid.material.opacity=.38;
    this.axes = new THREE.AxesHelper(.12); this.scene.add(this.axes);
    this.resize = new ResizeObserver(() => {
      const {width,height} = container.getBoundingClientRect();
      this.renderer.setSize(Math.max(width,1),Math.max(height,1));
      this.camera.aspect=width/Math.max(height,1);this.camera.updateProjectionMatrix();
    }); this.resize.observe(container);
    this.generation=0;
  }
  clear() {
    while(this.content.children.length) {
      const x=this.content.children[0];this.content.remove(x);
      x.traverse(o=>{o.geometry?.dispose(); if(o.material) (Array.isArray(o.material)?o.material:[o.material]).forEach(m=>m.dispose());});
    }
  }
  async load(episode, hand, variant, faces) {
    const generation=++this.generation;
    const meshSpec=variant.mesh||episode.mesh;
    const [vertices,indices,hands] = await Promise.all([loadBuffer(meshSpec.vertices),loadBuffer(meshSpec.faces),loadBuffer(hand.geometry)]);
    if(generation!==this.generation)return;
    this.clear();this.episode=episode;this.hand=hand;this.variant=variant;this.handBuffer=hands;
    const geometry = new THREE.BufferGeometry();
    geometry.setAttribute('position',new THREE.BufferAttribute(vertices,3));
    geometry.setIndex(new THREE.BufferAttribute(indices,1));geometry.computeVertexNormals();
    this.object = new THREE.Mesh(geometry,new THREE.MeshStandardMaterial({color:0xf5bc7c,roughness:.7,metalness:.05,side:THREE.DoubleSide}));this.content.add(this.object);
    this.handMeshes=['left','right'].map((side,j)=>{
      const geom=new THREE.BufferGeometry();
      geom.setAttribute('position',new THREE.BufferAttribute(new Float32Array(778*3),3).setUsage(THREE.DynamicDrawUsage));
      geom.setIndex(side==='left'?faces.flatMap(f=>[f[0],f[2],f[1]]):faces.flat());
      const mesh=new THREE.Mesh(geom,new THREE.MeshStandardMaterial({color:j===0?0x65d5bf:0xb5a1f4,roughness:.65,metalness:.03,side:THREE.DoubleSide}));
      mesh.frustumCulled=false;this.content.add(mesh);return mesh;
    });
    this.cameraMarker=new THREE.Group();
    const pts=[0,0,0,-.025,-.018,.045,0,0,0,.025,-.018,.045,0,0,0,-.025,.018,.045,0,0,0,.025,.018,.045,-.025,-.018,.045,.025,-.018,.045,.025,-.018,.045,.025,.018,.045,.025,.018,.045,-.025,.018,.045,-.025,.018,.045,-.025,-.018,.045];
    const cg=new THREE.BufferGeometry();cg.setAttribute('position',new THREE.Float32BufferAttribute(pts,3));
    this.cameraMarker.add(new THREE.LineSegments(cg,new THREE.LineBasicMaterial({color:0x637f98})));this.content.add(this.cameraMarker);
    const path=[];
    for(let i=0;i<variant.camera.length;i++){
      if(!variant.object_valid[i])continue;
      const t=matrix(variant.camera[i]).multiply(matrix(variant.object_camera[i]));path.push(new THREE.Vector3().setFromMatrixPosition(t));
    }
    this.predictedPath=new THREE.Line(new THREE.BufferGeometry().setFromPoints(path),new THREE.LineBasicMaterial({color:0xeab67c,transparent:true,opacity:.45}));this.content.add(this.predictedPath);
    if(variant.alignment){
      const {rotation:r,translation:t}=variant.alignment;
      const inverse=matrix([[...r[0],t[0]],[...r[1],t[1]],[...r[2],t[2]],[0,0,0,1]]).invert();
      const points=[];
      // Segments never bridge invalid GT samples.
      for(let i=1;i<episode.ground_truth.poses.length;i++){
        if(!episode.ground_truth.valid[i-1]||!episode.ground_truth.valid[i])continue;
        for(const j of [i-1,i])points.push(new THREE.Vector3(...episode.ground_truth.poses[j].slice(0,3).map(row=>row[3])).applyMatrix4(inverse));
      }
      this.truthPath=new THREE.LineSegments(new THREE.BufferGeometry().setFromPoints(points),new THREE.LineBasicMaterial({color:0xd6e0e6,transparent:true,opacity:.8}));this.content.add(this.truthPath);
    }
    const points=path.slice();
    // Frame typical motion, without letting a 12 m wrist outlier shrink every
    // useful detail. Outliers remain in the geometry and can be fit explicitly.
    const [,hn,hv,hd]=hand.geometry.shape;
    for(let i=0;i<hn;i+=Math.max(1,Math.floor(hn/24))){
      for(let s=0;s<2;s++){
        const side=s?'right':'left';
        if(!hand[side].valid[i]||hand[side].camera_distance_m[i]>1.5)continue;
        const offset=(s*hn+i)*hv*hd;const center=new THREE.Vector3();
        for(let v=0;v<hv;v++)center.add(new THREE.Vector3(...hands.subarray(offset+v*3,offset+v*3+3)));
        center.divideScalar(hv).applyMatrix4(matrix(variant.camera[i]));points.push(center);
      }
    }
    if(!points.length)points.push(new THREE.Vector3());
    const coords=['x','y','z'].map(k=>points.map(p=>p[k]).sort((a,b)=>a-b));
    this.target=new THREE.Vector3(...coords.map(c=>c[Math.floor(c.length/2)]));
    this.span=Math.max(.85,...coords.map(c=>c[Math.floor((c.length-1)*.95)]-c[Math.floor(c.length*.05)]));
    this.grid.position.z=this.target.z-.13;this.axes.position.copy(this.target);this.axes.position.z=this.grid.position.z;
    this.reset();this.frame(0);
  }
  reset(){
    if(!this.target)return;
    this.camera.position.copy(this.target).add(new THREE.Vector3(.8,-1.25,.9).multiplyScalar(this.span));
    this.controls.target.copy(this.target);this.camera.lookAt(this.target);this.controls.update();
  }
  fitFrame(index){
    if(!this.handMeshes)return;
    const box=new THREE.Box3();this.content.updateMatrixWorld(true);
    for(const mesh of [this.object,...this.handMeshes]){
      if(!mesh.visible)continue;
      mesh.geometry.computeBoundingBox();box.union(mesh.geometry.boundingBox.clone().applyMatrix4(mesh.matrixWorld));
    }
    if(box.isEmpty())return;
    const target=box.getCenter(new THREE.Vector3()),size=box.getSize(new THREE.Vector3());
    const distance=Math.max(size.length(),.65)*1.5;
    const direction=this.camera.position.clone().sub(this.controls.target).normalize();
    this.controls.target.copy(target);this.camera.position.copy(target).addScaledVector(direction,distance);this.controls.update();
  }
  frame(index){
    if(!this.variant||!this.handMeshes)return;
    const i=Math.min(index,this.variant.camera.length-1);
    const c2w=matrix(this.variant.camera[i]);
    setMatrix(this.object,c2w.clone().multiply(matrix(this.variant.object_camera[i])));
    this.object.visible=!!this.variant.object_valid[i];setMatrix(this.cameraMarker,c2w);
    const [,n,v,d]=this.hand.geometry.shape;
    this.handMeshes.forEach((mesh,j)=>{
      const side=j?'right':'left';mesh.visible=!!this.hand[side].valid[i];
      if(!mesh.visible)return;
      const offset=(j*n+i)*v*d;
      mesh.geometry.attributes.position.array.set(this.handBuffer.subarray(offset,offset+v*d));
      mesh.geometry.attributes.position.needsUpdate=true;mesh.geometry.computeVertexNormals();
      setMatrix(mesh,c2w);
    });
  }
  setTruth(value){if(this.truthPath)this.truthPath.visible=value;}
  render(){this.controls.update();this.renderer.render(this.scene,this.camera);}
  copyCamera(other){this.camera.position.copy(other.camera.position);this.camera.quaternion.copy(other.camera.quaternion);this.controls.target.copy(other.controls.target);this.controls.update();}
}
