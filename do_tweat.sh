while true; do
	export TWEAT="/home/kmd/twitter_credentials"
	python3 use_streamer.py --nspass tweat 127.0.0.1 tweat covid19 "..."
done
