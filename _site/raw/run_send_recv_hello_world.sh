for i in {0..7}; do
	for j in {0..7}; do
		if [ $i -ne $j ]; then
			python3 send_recv_hello_world.py $i $j | grep Effective
		fi
	done
done
