#! /bin/bash

git add .
time=$(date)
git commit -m "$time"
git push origin master
