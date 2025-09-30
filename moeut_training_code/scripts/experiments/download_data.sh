echo "start download dataset"

while true; do
    python3 scripts/download_dataset.py

    sleep 5 # sleep before start again
done