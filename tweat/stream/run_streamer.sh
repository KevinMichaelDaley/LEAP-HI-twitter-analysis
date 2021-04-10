#!/bin/bash
trap processSigTerm SIGTERM
SOURCE="${BASH_SOURCE[0]}"
# While $SOURCE is a symlink, resolve it
while [ -h "$SOURCE" ]; do
     DIR="$( cd -P "$( dirname "$SOURCE" )" && pwd )"
     SOURCE="$( readlink "$SOURCE" )"
     # If $SOURCE was a relative symlink (so no "/" as prefix, need to resolve it relative to the symlink base directory
     [[ $SOURCE != /* ]] && SOURCE="$DIR/$SOURCE"
done
DIR="$( cd -P "$( dirname "$SOURCE" )" && pwd )"
processSigTerm() {
	kill -15 $CHILD_PID
}
if [ ! -d "$DIR/run_streamer" ]; then
	$DIR/../install_deps.sh "$DIR/run_streamer"
fi
source $DIR/run_streamer/bin/activate
python $DIR/streamer.py $1 $2 $3 $4 "$5"
set CHILD_PID=$!
deactivate
rm -rf $DIR/run_streamer
