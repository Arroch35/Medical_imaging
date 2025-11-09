per="002345_Aug1"

if 'Aug' in per:
    per=per.split('_')
    per =  str(int(per[0]))+'_'+per[1]
else:
    per = str(int(per))

print(per)