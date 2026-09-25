# Judge Q&A: honest answers

**Q1. Did you train YOLO?**
No. We use the pretrained YOLO11n model, trained on the public COCO dataset. The bottle, cup, phone and scissors are
detected because they are COCO classes; we map them to BAS equipment *stand-in* labels. We included a fine-tuning script
(`tools/train.py`) for when a real BAS dataset is available, but we have not trained a custom model.

**Q2. Where is your ISRO dataset?**
We don't have one. No ISRO, BAS or astronaut data was used. The prototype uses pretrained perception models and a
controlled demonstration environment. Our dataset strategy (docs/dataset-strategy.md) describes how real BAS data would be
collected, annotated and used for fine-tuning and validation.

**Q3. Is this a trained Human Activity Recognition model?**
No. It is AI-assisted HAR: neural networks do the perception (YOLO for objects and people, MediaPipe for hands), and
activities are recognised by explicit temporal reasoning and a state machine. A trained temporal activity model is
future work that needs labelled BAS sequences.

**Q4. Why YOLO + MediaPipe?**
Each is specialised and fast on a CPU. YOLO answers "what objects and people are present, and where"; MediaPipe gives
21 precise hand landmarks, which a general detector does not. Combining them gives us hand-to-object relationships
without training anything, and each part can be replaced separately later.

**Q5. How do you detect interaction?**
It is 2D vision-based interaction estimation. For each hand we take the fingertips and palm centre. If one lies inside
the object's slightly enlarged bounding box, that frame counts as CONTACT; if it is within about 1.6 hand-lengths, NEAR.
The thresholds scale with hand size. It is **not** physical contact detection: we have no depth, so a hand passing in
front of an object can look like contact.

**Q6. How do you reduce false positives?**
- **Detection persistence.** A detection must persist for several detector runs before we trust the object.
- **Interaction timing.**
  - An interaction needs 4 consecutive contact frames.
  - A release needs 8 frames with the hand visible and away.
  - Contacts shorter than 0.5 s are ignored.
  - Each object has a cooldown after a completed activity.
- **Zones.** A zone needs 3 frames to trigger and 1 s to clear.
- **Alert cooldown.** Every alert has a cooldown, so nothing fires every frame.
- **Stricter confidence** for classes that get confused with hands, such as the phone.
- **Tests.** Unit tests check that single-frame contacts are rejected.

**Q7. What happens if the hand is occluded?**
If MediaPipe loses the hand during an interaction, we hold the state and show "HAND TRACKING LOST". If the hand comes
back within 1.5 s the interaction simply continues. If not, it ends as "interrupted" and is **not** counted as a
completed activity. A lost hand also never triggers an alert by itself, and the unexpected-movement rule is suspended
while hands aren't reliably tracked.

**Q8. What happens if the object is not detected?**
The tracker keeps the last box for a short time and the dashboard shows "OBJECT TEMPORARILY LOST". An object hidden by
the hand during contact is treated as "picked up". If the object stays undetected long enough for the track to be
dropped, the interaction is logged as interrupted, not completed. If a stable object disappears without anyone
handling it, a "missing object" warning is raised.

**Q9. Why rule-based safety logic?**
Safety rules must be transparent, predictable and testable, and we have no labelled abnormal-event data to train on.
Rules such as "hand in restricted zone for 3 s gives CRITICAL" can be reviewed, configured and unit-tested. Learned anomaly
detection could be added later, alongside the rules rather than replacing them.

**Q10. Why not cloud AI?**
Latency, privacy and connectivity. On board, links are limited and alerts must be immediate. Everything runs locally:
the server listens only on localhost, frames stay in memory, no video is recorded or uploaded, and the models are stored
locally (the application never downloads anything at run time).

**Q11. Can it run on Jetson?**
We have **not** tested it on Jetson. The architecture is designed with edge deployment in mind and can be optimised
for Jetson-class hardware using compatible dependencies, TensorRT, FP16 inference and hardware-specific tuning. We
include an export script (`tools/export_edge.py`), but Jetson performance is unmeasured.

**Q12. Can it work in space?**
Not as-is. This is a ground prototype. Space use would need:
- real BAS data and validation
- space-qualified computing hardware
- a radiation- and thermal-appropriate design
- camera placement studies
- formal verification

We make no mission-ready or flight-ready claim.

**Q13. How would you train it with real BAS data?**
1. Record the real procedures, normal and abnormal, from the real camera positions.
2. Annotate the equipment boxes, actions, hand-object contacts and step boundaries.
3. Fine-tune the detector on the BAS equipment.
4. Tune or learn the interaction thresholds.
5. If enough sequences exist, train a temporal activity model.
6. Validate on held-out sessions across lighting, angles and operators, reporting precision, recall and false-alert rates.

The details are in docs/dataset-strategy.md.

**Q14. What is the current limitation?**
- Only 2D interaction estimates, from a single camera.
- Generic pretrained perception using stand-in objects.
- Lightweight tracking, which can swap IDs when identical objects cross.
- Results depend on camera angle and lighting.
- Frame-based thresholds depend on the laptop's FPS.
- No measured accuracy figures yet.
