#! /bin/bash
# add all dir to PYTHONPATH
# run this: '''bash config_path.sh dirname'''
# for example '''bash config_path.sh .'''
function read_dir(){
    for file in `ls $1`
    do
        if [ -d $1"/"$file ]
        then
            # echo $1"/"$file
            export PYTHONPATH=$PYTHONPATH:$1"/"$file
            read_dir $1"/"$file
        fi
    done
} 

read_dir $1
# echo $PYTHONPATH