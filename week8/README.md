# Week 8: ACT on ALOHA sim (transfer cube)

Gate, written before training:
- Success = failed which i expect more than 80%
- Attempts = 50, because it is hard to decide the behavior of traning with the less attempts 
- Pass if success rate >= 80 % at 100k steps, because this time we use all the data so it should pass minimum 80% of what it trying to do in first traning 

Result: 40k 74% / 60k 76% / 80k 66% / 100k 78% 

i just think first times it should pass 80% but we did not reach there after first 40k runs it does not improve the process so for me am taking this as fail and also the other things are i watch a video it is good but it fails rearly so we need to impove it i do check it does not have a loop issue it is some issue why it is not updating after some steps we may use more demos focusing on what points are our trained model failling.

so i belive more loop helps after we update or add more demos focusing where our trained model are missing things 

Next test: add demos at the failure positions, same 100k steps. If success is still rising at 100k, then test longer training.

so to identify which one cause to have a better trained model we need to do it in sequence otherwise it gona be hard to know and understand what causes the model trained good.