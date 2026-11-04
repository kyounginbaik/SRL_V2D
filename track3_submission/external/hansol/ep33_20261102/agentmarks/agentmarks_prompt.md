You are annotating video frames for a 3D reconstruction pipeline. Work only from the images listed below; do not open any other file or folder and do not use the network.

Folder: C:\Users\PC\AppData\Local\Temp\claude\e2b544f74791fa27\a859a94c-8e15-4b21-8987-6a7db3e50a70\scratchpad\agentcrops
Files: f{NNN}.jpg for the frame numbers {RANGE} (step 2). Each is a crop around a wooden spoon from one RGB camera (20 fps). A person holds and moves the spoon, sometimes stirring in a blue cup, sometimes resting it on a round wooden table. The small grey/black balls on the spoon are tracking markers; ignore them.

For every frame decide which way the spoon bowl's concave (scooping, inner) side faces relative to the camera:
- "out": the concave inner side of the bowl faces the camera (you look into the bowl).
- "in": the convex back of the bowl faces the camera (the concave side faces away).
- [dx, dy]: the bowl is seen roughly edge-on; give the unit image-plane direction the concave side points to, x to the right, y down (e.g. [0, 1] = concave side points down in the image).
- "unsure": the bowl is not visible, hidden by the hand or cup, or you cannot tell.

Cues: shading and highlights inside a concave bowl versus on a convex back, the visible rim, occlusion of the inner surface, and continuity with neighbouring frames (look at neighbours when a frame is ambiguous, but judge each frame from its own pixels where possible). Do not guess: use "unsure" when the image does not support a call.

Write a JSON file to C:\Users\PC\AppData\Local\Temp\claude\e2b544f74791fa27\a859a94c-8e15-4b21-8987-6a7db3e50a70\scratchpad\agentmarks_{TAG}.json mapping the frame number (as a string, no leading zeros) to {"mark": <one of the above>, "conf": "high"|"medium"|"low", "why": "<at most 12 words>"}. Cover every frame in your range. In your final reply give only the file path and the counts of out / in / edge-on / unsure.
