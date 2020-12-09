#!/bin/bash

#test whole messages in messages directory

cd /home/unittest

output_dir=data/results

#ls $PWD/data/messages/gfrn/*>flist

find ./data/messages/prod -type f -name "*.msg" | xargs realpath > flist

i=1

while read line

do 
	
echo "process message $i: $line..."

./test_one_message.bash $line $output_dir

i=$(( $i + 1 ))

done < flist

rm flist

