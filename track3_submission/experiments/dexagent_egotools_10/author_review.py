"""Transcribe this session's visual review into the upstream Stage 1 schema.

The descriptions below are manually authored from the generated contact sheets.
Captions are used only in task selection, not loaded by this transcription code.
"""
from pathlib import Path
import json

ROOT=Path(__file__).resolve().parent
TASKS={t['id']:t for t in json.loads((ROOT/'tasks.json').read_text())}

def obj(ident,name,kind,role,props,evidence,**extra):
    return dict(id=ident,name=name,type=kind,role=role,properties=props,evidence=evidence,**extra)

def step(hand,motion,action,refs,goal,times,left,right,**extra):
    return dict(acting_hand=hand,motion=motion,action=action,object_refs=refs,goal=goal,
                time_window_s=times,hand_roles={'left':left,'right':right},**extra)

def save(ident,summary,objects,steps,success,support='tabletop',notes=None):
    for i,s in enumerate(steps,1):s['index']=i
    doc=dict(task_prompt=TASKS[ident]['task'],task_summary=summary,
             environment=dict(support_surface=support,camera='Moving head-mounted rectified stereo-left camera.'),
             hands={'used':sorted({side for s in steps for side in (['left','right'] if s['acting_hand']=='both' else [s['acting_hand']])})},
             objects=objects,subgoals=steps,task_success=success,
             notes=(notes or [])+['Current-session Codex visual review; fixed upstream library; no example reuse.',
                                 'Unsupported functional/material outcomes must remain explicit and must not be replaced with pose-only success.'])
    path=ROOT/'reviewed_semantics'/(ident+'.json');path.parent.mkdir(exist_ok=True)
    path.write_text(json.dumps(doc,indent=2))

save('07_fold_paper','Both hands fold the white sheet twice, crease the folds and hold the folded sheet.',[
    obj('paper','white paper sheet','deformable','manipulated',[
        'Thin sheet bends and folds rather than moving as one rigid solid.',
        'A crease remains after the fingers press along the fold.',
        'Two hands align the edges and fold the sheet into a smaller layered form.'],
        'Sheet is open at 0 s, folded near 1-4 s and folded again near 5-11 s.',
        deformation={'kind':'creased sheet','how_the_task_deforms_it':'The hands bend the sheet along straight creases and press the layers together.'})
], [
    step('both','hold','Hold the paper by its edges.',['paper'],'The paper is held in both hands.',[0,1],
         'Holds the left edge.','Holds the right edge.'),
    step('both','fold','Fold the sheet and press the first crease.',['paper'],'The paper is folded along a crease.',[1,4.8],
         'Aligns and supports the paper.','Turns the panel over and presses the crease.'),
    step('both','fold','Fold and crease the paper again.',['paper'],'The paper is folded twice with the creases retained.',[4.8,13.6],
         'Aligns the layered edges.','Folds the smaller sheet and presses the layers.'),
    step('left','hold','Hold the folded paper while the right hand moves aside.',['paper'],'The paper is held in the left hand.',[13.6,15.55],
         'Holds the folded sheet.','Reaches away from the paper.')
], 'The paper is folded twice with the creases retained, the paper is held in the left hand.',
notes=['The right-hand reach toward scissors at the end is outside the requested folding task.'])

save('10_crack_walnut','The left hand holds the red-handled nutcracker while the right hand places and stabilizes a walnut in its jaws, then the handles close.',[
    obj('nutcracker','red handled metal nutcracker','articulated','tool',[
        'Two handles and jaws pivot about a hinge.',
        'Closing the handles compresses the walnut between the metal jaws.',
        'The tool must not be represented as a rigid shape with immobile jaws.'],
        'Red handles are open at 0-2 s and squeezed around the walnut at 2-7 s.',
        parts=['jaw_a','jaw_b'],articulation={'moving_part':'jaw_a','fixed_part':'jaw_b','joint_type':'revolute',
        'axis':'Across the hinge pin','state_over_video':'Open, closes against the walnut, then opens.'}),
    obj('walnut','brown walnut','other','manipulated',[
        'Brittle shell surrounds an interior kernel.',
        'Sufficient jaw compression should fracture the shell.',
        'Shell fracture is a material change, not just a change of walnut pose.'],
        'Right hand holds the walnut at 0-1 s, positions it in the jaws near 1-3 s and holds it again near 8-9 s.')
], [
    step('left','hold','Hold the nutcracker with the jaws open.',['nutcracker'],'The nutcracker is held in the left hand.',[0,1.2],
         'Holds the red handles.','Holds the walnut.'),
    step('right','insert','Place the walnut between the nutcracker jaws.',['walnut','nutcracker'],
         'The walnut is between the nutcracker jaws.',[1.2,3.5], 'Supports the tool.','Positions and steadies the walnut.'),
    step('both','press','Squeeze the nutcracker handles against the walnut.',['nutcracker','walnut'],
         'The walnut shell is cracked by the nutcracker.',[3.5,7.8], 'Squeezes the tool handles.','Steadies the walnut at the jaws.',
         observed_outcome='Jaw closure is visible, but shell fracture is not resolved confidently in this short clip.'),
    step('left','place','Return the nutcracker to the mat.',['nutcracker'],'The nutcracker rests on the table.',[7.8,9],
         'Lowers and releases the tool.','Continues to hold the walnut.')
], 'The walnut shell is cracked by the nutcracker, the nutcracker rests on the table.',
support='dark mat on the table',notes=['Do not infer an observed successful shell fracture solely from the handle squeeze.'])

save('03_cut_paper','The left hand supports folded white paper while the right hand picks up orange-handled shears and repeatedly snips the edge.',[
    obj('paper','folded white paper','deformable','manipulated',[
        'Thin folded sheet bends while held in the left hand.',
        'Opposed shear blades must cut through the paper layers.',
        'The cut creates a slit from the edge; it does not require severing the entire sheet into two pieces.'],
        'Paper is held at 0-23 s; repeated snips enter its edge near 3-15 s and 19-23 s.',
        deformation={'kind':'folded cuttable sheet','how_the_task_deforms_it':'The sheet bends during handling and is cut locally by the shears.'}),
    obj('shears','orange handled shears','articulated','tool',[
        'Two metal blades pivot relative to each other.',
        'Finger pressure on the handles closes the blades through the paper.',
        'Blade closure must produce a material cut in the paper.'],
        'The right hand picks up the shears at 0-2 s and repeatedly closes them at 3-23 s.',
        parts=['blade_a','blade_b'],articulation={'moving_part':'blade_a','fixed_part':'blade_b',
        'joint_type':'revolute','axis':'Across the blade pivot','state_over_video':'Blades repeatedly open and close.'})
], [
    step('left','hold','Hold the folded paper.',['paper'],'The paper is held in the left hand.',[0,1],
         'Supports the folded sheet.','Reaches toward the shears.'),
    step('right','grasp_lift','Pick up the shears.',['shears'],'The shears are held in the right hand.',[0,2],
         'Keeps holding the sheet.','Closes around the orange handles and lifts.'),
    step('both','cut','Repeatedly snip into the folded paper edge.',['shears','paper'],
         'The paper has a slit cut from its edge by the shears.',[2,23],
         'Turns and supports the folded sheet.','Repeatedly closes the blades through the edge.')
], 'The paper has a slit cut from its edge by the shears, the paper is held in the left hand.')

save('01_grasp_test_tube','The right hand reaches to the rack, lifts a clear test tube and holds it above the table; the left hand approaches to inspect it.',[
    obj('test_tube','transparent test tube','rigid','manipulated',[
        'A slender hollow tube that remains shape-preserving while lifted.',
        'Can be grasped between fingers without an attached supporting joint.',
        'Must be lifted clear of its support and remain held.'],
        'The hand reaches toward the rack near 15-16 s and visibly holds the raised tube at 16-18.1 s.'),
    obj('rack','red rack','rigid','target',[
        'A slotted support for upright tubes.',
        'The tube can be withdrawn upwards without carrying the rack.',
        'The rack remains on the tabletop.'],
        'Red rack is visible on the table at 0 s and 8-18.1 s.')
], [
    step('right','grasp_lift','Grasp and lift the test tube from the rack.',['test_tube','rack'],
         'The test tube is held in the right hand, the test tube is held clear of the table.',[15,16.8],
         'Rests near the table edge.','Pinches the tube and lifts it.'),
    step('right','hold','Hold the tube for inspection.',['test_tube'],
         'The test tube is held in the right hand.',[16.8,18.1],
         'Approaches the lower end to assist inspection.','Continues to hold the tube.')
], 'The test tube is held in the right hand, the test tube is held clear of the table, the rack has not moved.',
notes=['The first fifteen seconds show positioning and a large head turn before the actual pickup.'])

save('05_pipette_transfer','The right hand repeatedly moves the pipette between the clear cup and the pink solution while the left hand rests.',[
    obj('pipette','clear plastic pipette','deformable','tool',[
        'A slender tube with a squeezable bulb and narrow dispensing tip.',
        'Bulb compression and release control liquid aspiration and dispensing.',
        'Liquid transfer requires more than positioning the tip over a cup.'],
        'The right hand holds the pipette and alternates between the cups at 0-13 s.',
        deformation={'kind':'compliant bulb','how_the_task_deforms_it':'The fingers squeeze and release the bulb during transfer.'}),
    obj('clear_cup','clear source cup','rigid','container',[
        'Open container supports liquid and permits pipette insertion.',
        'Stays on the tabletop beside the pink cup.',
        'The pipette tip is lowered into its interior.'],
        'Small clear cup is to the right of the pink cup at 0-13 s.'),
    obj('pink_cup','pink solution container','rigid','container',[
        'Open container receiving dispensed liquid.',
        'Its rim must not be treated as a closed solid top.',
        'Rests on the white paper below the hands.'],
        'The pink solution container remains on the white sheet at 0-13 s.'),
    obj('liquid','clear liquid','other','manipulated',[
        'Liquid is aspirated from the source cup and dispensed into the pink solution.',
        'Successful transfer requires an actual amount to leave the source and enter the destination.',
        'The amount transferred is not resolved from the overview images.'],
        'Pipette is inserted and moved between the source and destination at 0-13 s.')
], [
    step('right','hold','Hold the pipette.',['pipette'],'The pipette is held in the right hand.',[0,0.6],
         'Rests beside the torso.','Holds the pipette bulb.'),
    step('right','other','Draw liquid from the clear cup.',['pipette','clear_cup','liquid'],
         'The pipette contains liquid drawn from the clear source cup.',[0.6,4.8],
         'Remains resting.','Lowers the tip and operates the bulb.',
         observed_outcome='Insertion and movement are visible; liquid volume in the pipette is not resolved.'),
    step('right','other','Dispense into the pink solution container and repeat the transfer.',['pipette','pink_cup','clear_cup','liquid'],
         'Clear liquid has been transferred into the pink solution container.',[4.8,13],
         'Remains resting.','Alternates the pipette between the cups and squeezes the bulb.',
         observed_outcome='Tip movement between cups is visible, but the transferred liquid volume cannot be measured from these frames.')
], 'Clear liquid has been transferred into the pink solution container, the pipette is held in the right hand.')

save('08_wipe_counter','The right hand presses a white paper towel against the counter around the sink and makes repeated wiping strokes.',[
    obj('paper_towel','white paper towel','deformable','tool',[
        'Soft sheet conforms to the fingers and counter surface.',
        'Wiping requires sustained contact while the towel moves across the surface.',
        'A held towel above the counter does not establish a completed wiping action.'],
        'White towel is held against the counter in repeated strokes at 0-16 s.',
        deformation={'kind':'flexible sheet','how_the_task_deforms_it':'The hand bunches the towel and presses it onto the counter.'}),
    obj('counter','white kitchen counter','rigid','target',[
        'Fixed surface around a recessed sink.',
        'The target area extends along the right and front edges of the sink.',
        'Wiping should be evaluated from contact and coverage over this area.'],
        'White surface around the sink is visible throughout 0-17.5 s.')
], [
    step('right','hold','Hold the paper towel.',['paper_towel'],'The paper towel is held in the right hand.',[0,0.4],
         'Rests at the left side.','Holds the bunched towel.'),
    step('right','wipe','Wipe repeatedly around the sink.',['paper_towel','counter'],
         'The paper towel traverses the counter surface while pressed against it.',[0.4,16.6],
         'Remains resting.','Presses the towel against the counter and makes repeated strokes.',
         observed_outcome='Contact and repeated strokes are visible; hygienic cleanliness is not measurable from these images.')
], 'The counter surface has been wiped by the paper towel.',support='kitchen cabinetry',
notes=['The counter is an object because it is the explicit target of wiping; do not omit the contact-and-coverage requirement.'])

save('09_slice_banana','The left hand steadies a banana segment on the counter while the right hand repeatedly slices it with a kitchen knife.',[
    obj('banana','yellow banana segment','other','manipulated',[
        'Food segment can be separated into slices by a sharp blade.',
        'The left hand stabilizes the segment during the cuts.',
        'Successful slicing requires new separate pieces, not just a blade pose.'],
        'Banana is steadied at 1.5-3 s and cut repeatedly near 3-17.5 s.'),
    obj('knife','kitchen knife','rigid','tool',[
        'Long rigid blade and handle move together.',
        'The blade edge is lowered repeatedly through the banana.',
        'The cutting contact must cause material separation.'],
        'Right hand holds the knife throughout 0-17.5 s.')
], [
    step('right','hold','Hold the knife above the counter.',['knife'],'The knife is held in the right hand.',[0,1.6],
         'Moves toward the banana.','Holds the knife handle.'),
    step('left','hold','Steady the banana.',['banana'],'The banana is held in the left hand.',[1.6,3],
         'Stabilizes the banana on the counter.','Positions the blade.'),
    step('both','cut','Repeatedly slice the banana.',['knife','banana'],
         'The banana is cut into separate slices.',[3,17.5],
         'Keeps the uncut segment steady.','Lowers and raises the blade repeatedly.')
], 'The banana is cut into separate slices, the knife is held in the right hand.',support='kitchen counter',
notes=['The precise number and thickness of the small slices cannot be counted confidently from the overview.'])

save('04_stir_liquid','The left hand steadies and lifts a glass mug while the right hand inserts a green stirring rod, makes repeated circular strokes, and withdraws it.',[
    obj('stirring_rod','green stirring rod','rigid','tool',[
        'Slender shape-preserving rod held at its upper end.',
        'Its lower tip must enter the liquid inside the mug.',
        'Stirring requires a repeated path through the liquid, rather than a stationary inserted rod.'],
        'Right hand holds the green rod at 0-12.5 s and moves it inside the mug near 2-8 s.',parts=['tip']),
    obj('mug','transparent glass mug','rigid','container',[
        'Open cylindrical vessel with a handle.',
        'Contains liquid that is stirred by the rod.',
        'Left hand supports the mug while the right hand stirs.'],
        'Left hand lifts the mug near 1-2 s, holds it at 2-11 s and lowers it near 11-12.5 s.'),
    obj('liquid','liquid','other','manipulated',[
        'Fluid inside the mug must be contacted and agitated by the rod.',
        'Rod motion should be measured over time, not just at the final frame.',
        'No chemical reaction or quantitative homogeneity is established by the video.'],
        'The interior of the mug is visible while the rod moves in it at 2-8 s.')
], [
    step('left','grasp_lift','Lift and steady the mug.',['mug'],'The mug is held in the left hand.',[0.8,2.3],
         'Supports the mug near its rim.','Holds the green rod.'),
    step('right','insert','Insert the rod tip into the mug.',['stirring_rod:tip','mug'],
         'The stirring rod tip is inside the mug.',[1.5,2.8], 'Steadies the mug.','Lowers the rod tip.'),
    step('right','other','Stir with repeated circular strokes.',['stirring_rod','mug','liquid'],
         'The stirring rod tip traces repeated circular strokes through the liquid.',[2.8,8.2],
         'Continues holding the mug.','Moves the green rod repeatedly around the mug interior.'),
    step('right','hold','Withdraw and hold the stirring rod.',['stirring_rod'],
         'The stirring rod is held in the right hand.',[8.2,11.6], 'Supports the mug.','Lifts the rod tip out of the mug.'),
    step('left','place','Lower the mug to the table.',['mug'],'The mug rests on the table.',[11.6,12.5],
         'Lowers the mug.','Keeps holding the rod.')
], 'The liquid has been stirred with the stirring rod, the stirring rod is held in the right hand, the mug rests on the table.')

save('06_pour_liquid','The left hand holds a small clear container while the right hand tips a white plastic cup toward its opening to pour.',[
    obj('source_cup','white plastic cup','rigid','manipulated',[
        'Open vessel that can retain and pour liquid.',
        'The right hand tilts its rim above the receiving opening.',
        'The cup itself remains a rigid object.'],
        'Right hand holds and tilts the white cup throughout 0-11 s.'),
    obj('receiver','small transparent container','rigid','container',[
        'Small open container held by the left hand.',
        'Its opening receives poured liquid.',
        'Liquid must remain inside rather than only pass near its opening.'],
        'Left hand holds the transparent receiver below the source cup at 0-11 s.'),
    obj('liquid','liquid','other','manipulated',[
        'Fluid moves under gravity when the cup is tilted.',
        'A successful pour requires actual transfer into the receiver.',
        'Liquid volume cannot be established from cup orientation alone.'],
        'Cup is tipped over the receiver at 0-11 s; a small clear stream is difficult to resolve at this resolution.')
], [
    step('left','hold','Hold the receiving container.',['receiver'],
         'The small transparent container is held in the left hand.',[0,0.6],
         'Holds the receiver below the source cup.','Holds the source cup.'),
    step('right','hold','Hold the pouring cup.',['source_cup'],'The white plastic cup is held in the right hand.',[0,0.6],
         'Steadies the receiver.','Holds the white cup.'),
    step('both','pour','Tilt the cup and pour liquid into the receiver.',['source_cup','receiver','liquid'],
         'Liquid has transferred from the white plastic cup into the small transparent container.',[0.6,11],
         'Keeps the receiver under the pouring rim.','Tilts the white cup toward the receiver.',
         observed_outcome='The pouring alignment and tilt are visible; transferred liquid volume is not quantitatively resolved.')
], 'Liquid has transferred into the small transparent container, the small transparent container is held in the left hand, the white plastic cup is held in the right hand.')
