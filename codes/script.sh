#!/bin/bash

fun=('XC_DI' 'XC_DA_DI' 'XC_DA_DDI' 'XC_DI_DI')
inp=('iPSC' 'iPSC_timeshifted' 'Partial' 'Partial_timeshifted')

for f in "${fun[@]}";do
	for i in "${inp[@]}";do
		echo "$f" "$i"
		./TS_fit.py -t 40 -f $f -i $i	
	done
done
