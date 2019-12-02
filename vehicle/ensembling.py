import csv

sub_files = ['./best_submissions/resnet50_88.csv',
             './best_submissions/resnext50_90.csv',
             './best_submissions/wideresnet50_90.csv']

sub_weight = [1.5, 2, 2]  # Weights of the individual subs

Hlabel = 'Id'
Htarget = 'Category'
npt = 1  # number of places in target

place_weights = {}
for i in range(npt):
    place_weights[i] = 1/(i+1)

lg = len(sub_files)
sub = [None]*lg
for i, file in enumerate( sub_files ):
    # input files
    print("Reading {}: w={} - {}". format(i, sub_weight[i], file))
    reader = csv.DictReader(open(file,"r"))
    sub[i] = sorted(reader, key=lambda d: float(d[Hlabel]))

# output file
out = open("./best_submissions/_submission_ens.csv", "w", newline='')
writer = csv.writer(out)
writer.writerow([Hlabel,Htarget])

for p, row in enumerate(sub[0]):
    target_weight = {}
    for s in range(lg):
        row1 = sub[s][p]
        for ind, trgt in enumerate(row1[Htarget].split(' ')):
            target_weight[trgt] = target_weight.get(trgt,0) + (place_weights[ind]*sub_weight[s])
    tops_trgt = sorted(target_weight, key=target_weight.get, reverse=True)[:npt]
    writer.writerow([row1[Hlabel], " ".join(tops_trgt)])
out.close()
