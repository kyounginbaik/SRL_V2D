"""Validate native convex distance witnesses before using geometry gradients.

MuJoCo mesh distances can report zero despite separated witness points. We
accept a repaired positive distance only with a convex support certificate.
Otherwise solve the six-variable closest-points convex QP on unchanged hulls.
This is an offline geometry query; no physical collision model is replaced.
"""
import numpy as np
from scipy.optimize import minimize,LinearConstraint
from scipy.spatial import ConvexHull


class ConvexDistanceQuery:
    def __init__(self,model,raw_distance=None):
        import mujoco
        self.model=model;self.raw=raw_distance or mujoco.mj_geomDistance
        self.local={};self.hulls={};self.repairs=0;self.fallbacks=0;self.penetration_bounds=0;self.cache={};self.box_cache={}
        for geom in range(model.ngeom):
            if model.geom_type[geom]!=mujoco.mjtGeom.mjGEOM_MESH:continue
            mesh=int(model.geom_dataid[geom]);a=int(model.mesh_vertadr[mesh]);n=int(model.mesh_vertnum[mesh])
            vertices=np.asarray(model.mesh_vert[a:a+n],float).copy()
            hull=ConvexHull(vertices)
            self.local[geom]=vertices;self.hulls[geom]=hull.equations.copy()

    def update(self,data):
        self.data=data;self.cache={};self.box_cache={}

    def vertices(self,geom):
        if geom not in self.cache:
            r=np.asarray(self.data.geom_xmat[geom]).reshape(3,3);p=self.data.geom_xpos[geom]
            self.cache[geom]=self.local[geom]@r.T+p
        return self.cache[geom]

    def lower_bound(self,g1,g2):
        if g1 not in self.local or g2 not in self.local:return 0.
        for geom in (g1,g2):
            if geom not in self.box_cache:
                points=self.vertices(geom)
                self.box_cache[geom]=(points.min(0),points.max(0))
        a,b=self.box_cache[g1],self.box_cache[g2]
        return float(np.linalg.norm(np.maximum(np.maximum(b[0]-a[1],a[0]-b[1]),0.)))

    def distance(self,g1,g2,distmax=.25):
        m,d=self.model,self.data;points=np.zeros(6)
        distance=float(self.raw(m,d,int(g1),int(g2),distmax,points))
        if g1 not in self.local or g2 not in self.local:return distance,points
        va,vb=self.vertices(g1),self.vertices(g2)
        vector=points[3:]-points[:3];length=float(np.linalg.norm(vector))
        tol=2e-7
        if length>tol:
            u=vector/length
            # A supporting witness on each convex hull certifies the shortest
            # Euclidean segment. A negative penetration witness is handled by
            # the native collision query and the independent depth residual.
            certificate=(float(np.max(va@u))<=float(points[:3]@u)+tol and
                         float(np.min(vb@u))>=float(points[3:]@u)-tol)
            if certificate:
                if abs(distance-length)>tol:self.repairs+=1
                return length,points
            if distance<0 and abs(length+distance)<tol:return distance,points
        # A positive distance with contradictory/no witnesses must not create
        # a fake zero-gap contact at the origin. Solve the convex hull QP.
        self.fallbacks+=1
        matrices=[];bounds=[]
        for i,g in enumerate((g1,g2)):
            eq=self.hulls[g];r=np.asarray(d.geom_xmat[g]).reshape(3,3);p=d.geom_xpos[g]
            normal=eq[:,:3]@r.T;offset=eq[:,3]-normal@p
            block=np.zeros((len(normal),6));block[:,3*i:3*i+3]=normal
            matrices.append(block);bounds.append(-offset)
        constraint=LinearConstraint(np.concatenate(matrices),-np.inf,np.concatenate(bounds))
        center=.5*(va.mean(0)+vb.mean(0));size=max(float(np.ptp(np.r_[va,vb],axis=0).max()),.001)
        # Normalize coordinates to avoid prematurely stopping on mm-scale QPs.
        initial=(np.r_[va.mean(0),vb.mean(0)]-np.tile(center,2))/size
        cm=constraint.A*size;ub=constraint.ub-constraint.A@np.tile(center,2)
        def objective(x):return .5*np.sum((x[:3]-x[3:])**2)
        def gradient(x):delta=x[:3]-x[3:];return np.r_[delta,-delta]
        result=minimize(objective,initial,jac=gradient,constraints=[LinearConstraint(cm,-np.inf,ub)],
            method='SLSQP',options=dict(ftol=1e-13,maxiter=100))
        witness=result.x*size+np.tile(center,2);length=float(np.linalg.norm(witness[3:]-witness[:3]))
        if (not result.success or np.max(constraint.A@witness-constraint.ub)>tol
                or not np.isfinite(witness).all()):
            raise ValueError('convex closest-point query failed independent feasibility check')
        if length>tol:return length,witness
        # A zero QP distance certifies intersection, but not penetration depth.
        # Use the deepest native contact if available, with explicit witnesses.
        matches=[c for c in d.contact[:d.ncon] if set(map(int,c.geom))=={int(g1),int(g2)}]
        if matches:
            c=min(matches,key=lambda c:c.dist);normal=c.frame.reshape(3,3)[0].copy()
            if int(c.geom[0])!=int(g1):normal=-normal
            gap=float(c.dist);point=c.pos.copy()
            return gap,np.r_[point-.5*gap*normal,point+.5*gap*normal]
        # Face-normal separating axes give a conservative penetration upper
        # bound when native collision filtering has no pair record. A positive
        # separation is still resolved by the convex QP above. This bound is
        # a planning penalty, not a certified native contact or force patch.
        directions=[]
        for geom in (g1,g2):
            r=np.asarray(d.geom_xmat[geom]).reshape(3,3)
            directions.append(self.hulls[geom][:,:3]@r.T)
        axes=np.concatenate(directions);axes/=np.linalg.norm(axes,axis=1)[:,None]
        pa,pb=va@axes.T,vb@axes.T
        gap_ab=pb.min(0)-pa.max(0);gap_ba=pa.min(0)-pb.max(0)
        gaps=np.r_[gap_ab,gap_ba];index=int(np.argmax(gaps));gap=float(gaps[index])
        axis=axes[index%len(axes)]*(1 if index<len(axes) else -1)
        if gap>tol:raise ValueError('convex QP contradicts a separating-plane certificate')
        point_b=vb[int(np.argmin(vb@axis))]
        self.penetration_bounds+=1
        if gap>=-tol:gap=-tol
        return gap,np.r_[point_b-gap*axis,point_b]
