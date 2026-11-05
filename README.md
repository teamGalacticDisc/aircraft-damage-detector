# Aircraft Damage Detector

A computer vision model that finds cracks and dents in photos of aircraft skin. I built it for my ARIN 320 capstone to help maintainers spot damage faster. The model only points at spots worth checking. A human inspector still looks at every spot and makes the final call.

![Crack found at epoch 1, 10 and 20](results/images/crack_epoch_1_10_20.jpg)

## Why I built this

I work around F/A-18s. When you inspect an aircraft, you are looking at a lot of metal, and a small crack or dent is easy to miss when you are tired or rushed. Missing one can hurt a pilot or a maintainer. So I wanted a helper that looks at a photo and says, check here.

Think of it like a friend with a highlighter. The friend reads the page and highlights the parts that look important. You still read those parts yourself and decide what they mean. The highlighter never signs the paperwork. The inspector does.

## What the model does

You give it a photo. It draws a box around anything that looks like a **crack** or a **dent** and gives a confidence number from 0 to 1. A higher number means the model is more sure.

That is all it does. It does not decide whether the aircraft can fly.

## The pictures it learned from

I used the Aircraft Damage Detection dataset from the youssef-donia-fhktl workspace on Roboflow, version 5 (the one named Extra Finetuning, current best). It has 3,174 photos, and people already marked every crack and dent in them, either with a box or by tracing an outline around it.

| Group | Photos | What it is for |
|---|---|---|
| Train | 2,574 | The model studies these, like homework |
| Validation | 452 | A practice quiz after every round of studying |
| Test | 148 | The final exam, saved until the very end |

**Why version 5?** My professor first gave us a different dataset, and a test run on it only reached a score of about 0.02, which means it could barely learn anything. The professor switched the class to this one because it reached about 0.58. Good pictures with good labels matter more than anything else.

**Why keep the test photos separate?** If you study the exam questions, a perfect score does not prove you learned anything. Keeping the test photos hidden until the end is supposed to show how the model does on photos it has never seen, which is what would happen on a real flight line.

**A catch I found in this dataset.** The dataset makers took each original photo and made several copies of it, a little brighter, darker or flipped, and those copies got spread across all three groups. For example, the same damaged wing (photo 158) shows up 3 times in training and 2 times in validation. All 99 original photos behind the validation pictures and all 37 behind the test pictures also show up in training. So the model has seen a close cousin of almost every quiz and exam photo. That means my scores are probably higher than what it would get on brand new photos from a real hangar. My professor's 0.58 used the same dataset, so comparing with it is still fair.

## How I trained it

I used **YOLOv8n**, the smallest YOLO model.

**Why YOLO?** YOLO stands for You Only Look Once. It looks at the whole photo one time and finds every object at once, so it is fast. Fast matters if a maintainer is walking around a jet taking pictures.

**Why the small one?** I trained it on a regular computer with no graphics card. The small model learns quickly and still runs fine on a laptop or tablet later.

**Why start from a pretrained model?** It has already seen millions of everyday photos, so it knows what edges, corners and shadows look like. It only has to learn what a crack and a dent look like. That is like teaching someone who already knows how to read a new word, instead of teaching the alphabet first.

**Why 20 epochs?** One epoch means the model looks at every training photo one time. My professor's reference run used 20, so I used 20 so my result could be compared with hers.

## How it got better

Every epoch the model took a practice quiz on the validation photos. The score here is **mAP50**. In plain words, it asks: when the model drew a box, was it in the right spot and did it pick the right label? It goes from 0 (always wrong) to 1 (perfect).

| After epoch | mAP50 | What was happening |
|---|---|---|
| 1 | 0.05 | Guessing. Boxes everywhere. |
| 5 | 0.19 | Starting to notice the damage |
| 10 | 0.39 | Finding some cracks and dents, but still missing a lot |
| 15 | 0.55 | Boxes getting tighter and more accurate |
| 20 | **0.59** | Done. A little above the professor's 0.58 |

Look at the crack picture at the top. At epoch 1 the model drew a pile of messy boxes. By epoch 10 it found the crack with one box but was not very sure (0.45). By epoch 20 it was almost certain (0.98).

The dent below shows the same story. At epoch 1 it called a patch of sky a crack. By epoch 20 it found the real dent and called it a dent.

![Dent found at epoch 1, 10 and 20](results/images/dent_epoch_1_10_20.jpg)

This chart shows every epoch at once. The loss charts going down mean the model is making fewer mistakes. The mAP charts going up mean it is getting more right.

![Training chart](results/results.png)

## Final results

| | Validation | Test |
|---|---|---|
| mAP50 | 0.595 | 0.627 |
| Precision | 0.66 | 0.73 |
| Recall | 0.56 | 0.58 |

| Damage type | Validation mAP50 | Test mAP50 |
|---|---|---|
| Crack | 0.64 | 0.63 |
| Dent | 0.55 | 0.63 |

**Precision** answers: when the model says damage, how often is it right? About 7 times out of 10 on the test photos.

**Recall** answers: out of all the real damage, how much did the model find? A little more than half.

That recall number is the most important thing in this whole project. You can see it with your own eyes in the confusion matrix below. **The model misses about 4 out of every 10 damage spots.** That is exactly why it can only be a helper. If an inspector trusted it alone, a lot of damage would get missed. Used the right way, as a second set of eyes that points at places to look, it can still catch things a tired person might walk past.

## The confusion matrix

A confusion matrix is a scoreboard that shows every time the model was right, every time it was wrong, and what kind of wrong it was. I ran it on the 449 validation photos with the same setting `detect.py` uses: the model only counts a box if it is at least 25 percent sure.

![Confusion matrix](results/confusion_matrix.png)

How to read it: each **column** is the truth (what was really in the photo). Each **row** is what the model said. Background means nothing there.

| | Really a crack | Really a dent | Really nothing |
|---|---|---|---|
| Model said crack | **326** right | 0 | 141 false alarms |
| Model said dent | 0 | **291** right | 173 false alarms |
| Model said nothing | 200 missed | 191 missed | |

What this tells me in plain words:

* **It never mixed up a crack and a dent.** Those two boxes are 0. When it finds damage, it names it correctly.
* **It found 326 of 526 cracks (62 percent) and 291 of 482 dents (60 percent).** The other 391 spots it walked right past. That is the about 4 in 10 missed from above.
* **It raised 314 false alarms**, drawing a box where nothing was marked. I have not gone through each one, so I cannot say yet what fooled it. Some could even be real damage the person labeling the photos skipped.

The big lesson: the problem is not that the model gets confused. The problem is that it does not see everything. That is a lot safer for a helper tool, because a missed spot still gets looked at by the inspector who walks the whole aircraft anyway.

## Model guesses next to the answer key

The left picture is what the people who made the dataset marked. The right picture is what my model found on the same photos. You can spot both the hits and the misses yourself.

| Answer key | Model guesses |
|---|---|
| ![Answer key](results/images/validation_answer_key.jpg) | ![Model guesses](results/images/validation_predictions.jpg) |

## Precision and recall together

![Precision recall curve](results/precision_recall_curve.png)

This chart shows a trade. If you make the model more careful (only show boxes it is very sure about), you get fewer false alarms but it misses more damage. If you make it less careful, it finds more damage but cries wolf more often. The bigger the area under the line, the better the model. For an inspection helper I would rather it be a little less careful, because a false alarm costs an inspector a few seconds and a miss could cost a lot more.

## What to keep in mind

* It was trained on general aircraft photos from the internet, not F/A-18 inspection photos. Before anyone used something like this for real, it would need to be retrained and tested on photos from that aircraft.
* It only knows two things, cracks and dents. It has never been taught corrosion, missing fasteners or anything else.
* 67 photos in the dataset (62 training, 3 validation, 2 test) had their damage marked two different ways in the same label file, some with boxes and some with traced outlines. YOLO throws out any photo that mixes the two, so the model never learned from those 67 photos. That is why the results show 449 validation photos instead of 452 and 146 test photos instead of 148.
* 20 epochs is short. The scores were still climbing at the end, so more epochs would probably help.

## Try it yourself

```
pip install -r requirements.txt
python detect.py your_photo.jpg
```

The trained model is in `model/best.pt`. To train it again from scratch, download the dataset (version 5, YOLOv8 format), unzip it into a folder called `data` next to these files, and run:

```
python train.py
python evaluate.py
```

## Files

| File | What it is |
|---|---|
| `train.py` | Teaches the model |
| `evaluate.py` | Grades the model on validation and test photos |
| `detect.py` | Runs the model on your own photo |
| `model/best.pt` | The trained model |
| `results/results.csv` | The score for every epoch |
| `results/results.png` | Chart of all 20 epochs |
| `results/confusion_matrix.png` | Scoreboard of hits, misses and false alarms on the validation photos |
| `results/precision_recall_curve.png` | The trade between false alarms and misses |
| `results/images/` | Before and after pictures, plus model guesses next to the answer key |

## Credits

Dataset: Aircraft Damage Detection (Roboflow workspace youssef-donia-fhktl), version 5, Roboflow Universe, CC BY 4.0. https://universe.roboflow.com/youssef-donia-fhktl/aircraft-damage-detection-1j9qk/dataset/5

Photos in the dataset carry their original photographer credits (for example aviation-safety.net) in the corner of each image.

Model: YOLOv8 by Ultralytics.
