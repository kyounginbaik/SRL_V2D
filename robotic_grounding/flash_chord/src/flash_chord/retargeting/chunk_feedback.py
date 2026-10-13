"""Feedback grouped by the same interior-frame labels as chunk reference edits."""
import numpy as np

def chunk_labels(frame_ids, frames, fps, chunk_seconds):
    width=max(1,int(round(fps*chunk_seconds)))
    count=(frames-3)//width+1
    return np.clip((np.asarray(frame_ids,dtype=int)-1)//width,0,count-1),count

def summarize_chunks(residual, frame_ids, frames, fps, chunk_seconds):
    error=np.asarray(residual,dtype=np.float64)
    if error.ndim!=4 or error.shape[-1]!=3 or error.shape[2]%2:
        raise ValueError('Expected time x world x even-keypoints x xyz')
    ids=np.asarray(frame_ids).reshape(error.shape[:2])
    labels,count=chunk_labels(ids,frames,fps,chunk_seconds)
    error=np.clip(np.nan_to_num(error,nan=0.,posinf=1.,neginf=-1.),-1,1)
    hands=np.stack([error[:,:,:error.shape[2]//2].mean(2),error[:,:,error.shape[2]//2:].mean(2)],axis=2)
    result=np.zeros((count,2,2,3),np.float32);coverage=np.zeros(count,np.int32)
    for k in range(count):
        world_means=[]
        for w in range(error.shape[1]):
            mask=labels[:,w]==k;coverage[k]+=int(mask.sum())
            if mask.any():world_means.append(hands[mask,w].mean(0))
        if world_means:
            result[k,0]=np.mean(world_means,axis=0);result[k,1]=np.std(world_means,axis=0)
    return result,coverage

def correction_chunk_means(correction, fps, chunk_seconds):
    q=np.asarray(correction);ids=np.arange(1,len(q)-1)
    labels,count=chunk_labels(ids,len(q),fps,chunk_seconds)
    return np.stack([q[ids[labels==k]].mean(0) for k in range(count)])
