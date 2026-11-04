import sys,json,imageio,numpy as np
from PIL import Image
items=json.loads(sys.argv[1]);out=sys.argv[2];rs=[]
for path,first in items:
    r=imageio.get_reader(path);rs.append(([np.asarray(x) for x in r],first))
n=150;w=imageio.get_writer(out,fps=20,codec='libx264',quality=None,ffmpeg_params=['-crf','18','-preset','medium','-pix_fmt','yuv420p'],macro_block_size=None)
for f in range(n):
    for k in range(2):
        t=[]
        for fr,first in rs:
            i=2*f if first<0 or f<=first else 2*f+int(1.5*20/2*2)
            t.append(np.asarray(Image.fromarray(fr[min(i+k,len(fr)-1)]).resize((960,540),Image.LANCZOS)))
        w.append_data(np.vstack([np.hstack(t[:2]),np.hstack(t[2:])]))
w.close();print("wrote",out,flush=True)
