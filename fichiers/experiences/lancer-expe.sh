
export OMP_NUM_THREADS

if [ $# == 0 ]
then
  iter=1000
else
  iter=$0
fi

ITE=$(seq 10) # nombre de mesures

THREADS=$(seq 2 2 24) # nombre de threads

PARAM="-n -s 128 -a -i $iter" # parametres commun à toutes les executions

execute (){
  EXE="./prog $* $PARAM"
  OUTPUT="$(echo $EXE | tr -d ' ')"
  for nb in $ITE
  do
    for OMP_NUM_THREADS in $THREADS
    do
      echo -n "$OMP_NUM_THREADS " >> $OUTPUT
      $EXE 2>> $OUTPUT
    done
  done
}

execute -v 0
execute -v 1
execute -v 2
execute -v 3
