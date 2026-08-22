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
