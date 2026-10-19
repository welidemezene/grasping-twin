# Daily log

22026-08-21 | DSA: Top K, Encode/Decode Strings | shipped: week6 closed, lerobot running in WSL | sleep: 6h
2026-08-22 | DSA: product of array except self | shipped: rendered first real robot frame from svla_so101_pickplace | sleep: 6h

## Images
A picture is just numbers. PyTorch stores it as 3 stacked sheets
(all reds, all greens, all blues). Image files store it dot by dot
(R,G,B for each dot). permute(1,2,0) changes sheets into dots.
If the picture looks striped, the permute is wrong. If it's black,
I forgot to multiply by 255.

## Where am I
Read the prompt before typing. (lerobot) = venv on. woldemedihn@ = WSL.
>>> = Python. $ = bash. Four problems today were "wrong place".

## Python
Every time python runs it starts fresh. Variables do not survive.
python -c is for one lines only. Use a file and re-run it.
## Where am I
Read the prompt before typing. (lerobot) = venv on. woldemedihn@ = WSL.


what i learn today

we have a frames and we use the list data and draw it to see the real and pridicted one , and look on the difference gap we have which is a time it takes to start.


yesterday i solved in neetcode 75 in array and hashing the first three easy question.

then today i solved the longest conscutive number.
it is a middium question i solved by my self with in the first 30 minute it was nlogn(n) times but it should be o(n).

the reason it become nlog(n) is we use sort() then we iterate on them check two index at the same times have a cnt and also max then we resolve it but next am gona solve it in a o(n) times which i have to use other data types.

in today october 11 

i understand more better the concepts we have what each key word is and also we trying to understand with one number first before doing for a 42 number so i understand everythings what the sensor have what the command and also and also there is a things we need also got a prediction by multiplying the state and w and also we have an slopes to check the loss and also the gaps we have to move forward and also backward this is what we have been working for todays.





In your plot: blue = your model, orange = the human, green = the real gripper.

Q1. Between frames 170 and 220, the green line is stuck at 7, but your blue line is near 1, like the orange line.
→ How can your model guess 1 when the gripper sensor says 7?

when the arm is in this carrying position, the human is squeezing, so guess about 1." It used the other joints as clues, even though the gripper sensor said 7

Q2. Compare blue (your model) and green (the copy-cat) against orange (the human).
→ In which part of the plot is your model closer to the human than the copy-cat is?


Frames 170–220 are where your model beats the copy-cat by the most. The copy-cat is off by 6 there (7 vs 1), which squares to 36, for about 50 frames.


Q3. Around frame 222, the orange line goes up first, and the blue line goes up later.
→ Why does your model react late?

 The lag is part of it, but the main reason is this: the human decides to open the gripper in their head first.

Q4. Around frame 235, the blue line goes below 0, which a real gripper can't do.
→ Why can your model output a number that's impossible?
The learning rate only controls how fast training goes. The real reason: a linear model is just a weighted sum of numbers, with no limits. 




