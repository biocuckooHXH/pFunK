import scipy.stats as stats

go = {}
gog = {}
omics = 'kbhb'
f = open('./data/goa_mouse.gaf', 'r')
if (omics == "Transcriptome") or (omics == "score"):
    for l in f:
        sp = l.rstrip().split('\t')
        if 'UniProtKB' in sp:
            if sp[4] in go:
                go[sp[4]] = go[sp[4]] + [sp[2].lower()]
            else:
                go[sp[4]] = [sp[2].lower()]
            gog[sp[2].lower()] = ''
else:
    for l in f:
        sp = l.rstrip().split('\t')
        if 'UniProtKB' in sp:
            if sp[4] in go:
                go[sp[4]] = go[sp[4]] + [sp[1]]
            else:
                go[sp[4]] = [sp[1]]
            gog[sp[1]] = ''



goid = {}
gotp = {}
gid = ''
name = ''
tp = ''

f = open('./data/go.obo', 'r')
for l in f:
    l = l.rstrip()
    if l == '[Term]':
        if gid != '' and name != '' and tp != '':
            goid[gid] = name
            gotp[gid] = tp
        gid = ''
        name = ''
        tp = ''
    if l.startswith('id: GO:'):
        gid = l.split('id: ')[1]
    if l.startswith('name: '):
        name = l.split('name: ')[1][0].upper() + l.split('name: ')[1][1:]
    if l.startswith('namespace: '):
        tp = l.split('namespace: ')[1]

NN = {}
bgene = {}

import pandas as pd
if omics == 'Transcriptome':
    count_annot = pd.read_excel('./data/count.annot.xlsx')
    for l in count_annot['gene_name'].unique():
        id = l.rstrip().lower()
        if id in gog:
            NN[id] = ''
        bgene[id] = ''
elif omics == 'proteomics':
    proteomics_df = pd.read_excel("./data/pro_perseus_fs.xlsx")
    for l in proteomics_df['protein ID'].unique():
        id = l.rstrip()
        if id in gog:
            NN[id] = ''
        bgene[id] = ''


elif omics == 'kbhb':
    kbhb_perseus_fs = pd.read_excel('./data/kbhb_fs_0625.xlsx')
    for l in kbhb_perseus_fs['Protein accession'].unique():
        id = l.rstrip()
        if id in gog:
            NN[id] = ''
        bgene[id] = ''

else:
    count_annot = pd.read_excel('./data/count.annot.xlsx')
    proteomics_df = pd.read_excel("./data/pro_perseus_fs.xlsx")
    kbhb_perseus_fs = pd.read_excel('./data/kbhb_perseus_fs.xlsx')
    count_annot_list = [i.strip().lower() for i in count_annot['gene_name'].unique()]
    proteomics_df_list = [i.strip().lower() for i in proteomics_df['Gene name'].unique()]
    kbhb_df_list = [i.strip().lower() for i in kbhb_perseus_fs['Gene name'].unique()]
    protein_list = list(set(proteomics_df_list + kbhb_df_list))
    for l in protein_list:
        id = l.rstrip()
        if id in gog:
            NN[id] = ''
        bgene[id] = ''

uid = {}
MM = {}
gene = {}
if omics == 'Transcriptome':
    f = open('./data/control_treat.DESeq2.select_processed.txt', 'r')
    for l in f:
        sp = l.rstrip().split('\t')
        gene_name = sp[-1].lower()
        if gene_name not in bgene:
            print(l)
            bgene[gene_name] = ''
            if gene_name in gog:
                NN[gene_name] = ''
        if gene_name in gog:
            MM[gene_name] = ''

        gene[gene_name] = ''
        uid[gene_name] = sp[-1]

elif omics=='proteomics':
    proteomics_fold_change = pd.read_table('./data/proteomics_DA_log2_0.5.txt',sep='\t')
    for index, row in proteomics_fold_change.iterrows():
        protein_id = row['Protein accession']
        if protein_id not in bgene:
            bgene[protein_id] = ''
            if protein_id in gog:
                NN[protein_id] = ''
        if protein_id in gog:
            MM[protein_id] = ''
        gene[protein_id] = ''
        uid[protein_id] = row['Gene name']


elif omics=='kbhb':
    kbhb_fold_change = pd.read_table('./data/kbhb_DA_log2_1_0625.txt', sep='\t')
    for index, row in kbhb_fold_change.iterrows():
        protein_id = row['Protein accession']
        if protein_id not in bgene:
            bgene[protein_id] = ''
            if protein_id in gog:
                NN[protein_id] = ''
        if protein_id in gog:
            MM[protein_id] = ''
        gene[protein_id] = ''
        uid[protein_id] = row['Gene name']


else:
    # core_genes = []
    score_core = open("./data/network_analysis/0514/PPI_cut_off_all.txt", 'r')
    lines = score_core.readlines()
    for line in lines:
        gene_name = line.strip().lower()
        # sp = l.rstrip().split('\t')
        if gene_name not in bgene:
            print(l)
            bgene[gene_name] = ''
            if gene_name in gog:
                NN[gene_name] = ''
        if gene_name in gog:
            MM[gene_name] = ''

        gene[gene_name] = ''
        uid[gene_name] = gene_name


# print(gene.__len__(), bgene.__len__(), goid.__len__(), go.__len__(), MM.__len__(), NN.__len__())
# w = open('./data/enrichment Analysis/results/Kbhb_go_0625_1.txt'.format(omics), 'w')
# for gg in go:
#     mm, nn = {}, {}
#     mm1 = {}
#     for gx in MM:
#         if gx in go[gg]:
#             mm[gx] = ''
#             mm1[uid[gx]] = ''
#     for gx in NN:
#         if gx in go[gg]:
#             nn[gx] = ''
#     m = mm.__len__()
#     M = MM.__len__()
#     n = nn.__len__()
#     N = NN.__len__()
#
#     if n == 0:
#         continue
#
#     # m=12
#     # M=458
#     # n=868
#     # N=19335
#     p = stats.fisher_exact([[m, M - m], [n - m, N - n - M + m]])
#     print(p)
#     print(m / M, n / N, (m / M) / (n / N))
#     if gg not in goid.keys() or gg not in gotp.keys():
#         continue
#     else:
#         w.write(gg + '\t' + goid[gg] + '\t' + gotp[gg] + '\t' + str(m) + '\t' + str(M) + '\t' + str(m / M) + '\t' +
#                 str(n) + '\t' + str(N) + '\t' + str(n / N) + '\t' + str((m / M) / (n / N)) + '\t' + str(p[1]) +
#                 '\t' + ', '.join(mm1) + '\n')

# https://blog.csdn.net/Raider_zreo/article/details/102251418   www.douyu.com/533813   GO:0043312	neutrophil degranulation	40	458	0.09	481	19335	0.02	3.51	5.79e-12
# GO:0007186	G-protein coupled receptor signaling pathway	12	458	0.03	868	19335	0.04	0.58	0.03

