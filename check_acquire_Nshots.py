import acquire_Nshots
import epics
import pvaccess as pva

def check_pva_connection(pv_name):
    # Create a PVA client object
    pv = pvaccess.Pv(pv_name)
    
    pv.connect()
        
    # Check if the connection is established
    if pv.is_connected():
        print(f"PVA PV {pv_name} is connected.")
    else:
        print(f"\033[1;31mPVA PV {pv_name} is not connected.\033[0m")


if __name__ == "__main__":
	# scalars
	print("SCALARS\n=======")
	for i in range(len(acquire_Nshots.scalars)):
		pv = epics.PV(acquire_Nshots.scalars[i])
		
		# wait for connection to be established
		pv.wait_for_connection(timeout=5)
		
		if pv.connected:
			print(f"{acquire_Nshots.scalars[i]} is connected.")
		else:
			print(f"\033[1;31m{acquire_Nshots.scalars[i]} is not connected. \033[0m")

	# arrays
	print("\nARRAYS\n======")
	for arraypv in acquire_Nshots.arrays:
		pv = epics.PV(arraypv)
		# wait for connection to be established
		pv.wait_for_connection(timeout=5)
		
		if pv.connected:
			print(f"{arraypv} is connected.")
		else:
			print(f"\033[1;31m{arraypv} is not connected. \033[0m")

	# images
	print("\nIMAGES\n======")
	for pv in acquire_Nshots.images:
		try:
		    channel = pva.Channel(pv)
		    pva_image = channel.get('')
		    width  = pva_image['dimension'][0]['size']
		    height = pva_image['dimension'][1]['size']

		    print(f"{pv}: ({width},{height})")
		except Exception as e:
			print(f"\033[1;31m{pv}: Error - {e}.\033[0m")
