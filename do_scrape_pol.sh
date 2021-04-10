while true; do
    python3 scrape_pol.py pnd8 8kun.top pnd res >> pnd_8.txt
    sleep 1
    python3 scrape_pol.py pol4 a.4cdn.org pol thread >> pol_4.txt
    sleep 1
    python3 scrape_pol.py news4 a.4cdn.org news thread >> news_4.txt
    sleep 1
done
