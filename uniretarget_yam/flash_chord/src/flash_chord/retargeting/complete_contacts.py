"""Observe every physical hand collider without changing policy observations."""
from dataclasses import dataclass, replace

import numpy as np


def hand_contact_evidence(editor):
    """Prefer complete physical evidence; expose semantic-only fallback clearly."""
    complete = hasattr(editor, '_physical_contact_active')
    prefix = '_physical_contact' if complete else '_contact'
    return dict(active=getattr(editor, prefix + '_active'), force=getattr(editor, prefix + '_force'),
        position=getattr(editor, prefix + '_pos', None),
        moment_world_about_object_origin=getattr(editor, prefix + '_moment_world_about_object_origin', None),
        slot_hand=getattr(editor, prefix + '_slot_hand'), slot_object=getattr(editor, prefix + '_slot_object'),
        sides=getattr(editor, prefix + '_sides'), links=getattr(editor, prefix + '_slot_link', ()),
        complete=complete)


@dataclass
class CompleteHandContacts:
    tracker: object
    slot_links: tuple[str, ...]
    moment_observer: object = None

    @property
    def sides(self):
        return tuple(self.tracker.layout.sides)

    @property
    def slot_hand(self):
        return np.asarray(self.tracker.layout.slot_hand_ids, np.int32)

    @property
    def slot_object(self):
        return np.asarray(self.tracker.layout.slot_object_ids, np.int32)

    def update(self, env):
        """Read solved forces and POST-transported geometry; never run physics.

        Optional moments use exposed local points with this state's poses.
        They are not certified native force-evaluation torques when the solver
        cached those points before its final integration.
        """
        self.tracker.update(env.model, env.state_0, env.contacts)
        worlds, slots = env.world_count, self.tracker.layout.slots_per_world
        result = dict(active=self.tracker.contact_active.numpy().reshape(worlds, slots).copy(),
            force=self.tracker.contact_force_w.numpy().reshape(worlds, slots, 3).copy(),
            position=self.tracker.contact_pos_w.numpy().reshape(worlds, slots, 3).copy())
        if self.moment_observer is not None:
            self.moment_observer.update(env.model, env.state_0, env.contacts)
            result['moment_world_about_object_origin'] = (
                self.moment_observer.moment_world_about_object_origin.numpy().reshape(worlds, slots, 3).copy())
        return result


def build_complete_hand_contacts(env, *, tracker_factory=None, include_moments=False):
    """Build an independent tracker including nonsemantic knuckle colliders.

    Link ownership comes from the embodiment's complete collision geometry.
    Only the private observer's contact roles change. The environment's scene,
    collision filters, semantic tracker, actor observations and physics remain
    unchanged. This avoids mistaking an unobserved giver-knuckle hook for a
    completed release.

    ``include_moments`` additionally sums each solved raw ``r x F + tau`` ON
    OBJECT about its BODY origin in WORLD axes, with ``r`` to the midpoint of
    the two contact surface points. It never reconstructs torque from the
    aggregate position/force and is absent from the legacy default.
    """
    if tracker_factory is None:
        from flash_chord.runtime.contact import ContactTracker
        tracker_factory = ContactTracker
    original = env.scene.layout
    hands = tuple(replace(hand, link_geometry=tuple(replace(link, object_contact=True)
                                                   for link in hand.link_geometry)) for hand in original.hands)
    layout = replace(original, hands=hands)
    tracker = tracker_factory.build(layout, object_body_ids=env.command.layout.body_ids,
        world_count=env.world_count, bodies_per_world=env.command.bodies_per_world,
        shapes_per_world=env.scene.collision_layout.shape_count, config=env.config.contact,
        device=env.device)
    links = tuple(link.link_name for side in tracker.layout.sides
        for _ in range(tracker.layout.num_objects) for link in layout.hand(side).contact_links)
    if len(links) != tracker.layout.slots_per_world:
        raise ValueError('complete physical contact schema does not match the tracker')
    moment_observer = None
    if include_moments:
        from flash_chord.retargeting.contact_moments import ContactMomentObserver
        moment_observer = ContactMomentObserver.build(tracker, device=env.device)
    return CompleteHandContacts(tracker, links, moment_observer)
