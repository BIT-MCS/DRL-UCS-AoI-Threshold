ps -ef | grep ray | grep -v grep | awk '{print $2}' | xargs kill -9
ps -ef | grep mappo | grep -v grep | awk '{print $2}' | xargs kill -9