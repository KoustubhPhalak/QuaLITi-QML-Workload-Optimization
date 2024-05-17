'''This file contains code to obtain hardware queue data. Kindly replace token with
   user's token from their IBM Quantum profile.'''

from qiskit_ibm_runtime import QiskitRuntimeService
from datetime import datetime
 
# Save an IBM Quantum account and set it as your default account.
# QiskitRuntimeService.save_account(channel="ibm_quantum", token="xxx", set_as_default=True)
 
# Load saved credentials
service = QiskitRuntimeService()

# Load backends
brisbane = service.backend('ibm_brisbane')
osaka = service.backend('ibm_osaka')
kyoto = service.backend('ibm_kyoto')

# Load current date and time
now = datetime.now()
dt_string = now.strftime("%m/%d/%Y %H:%M")

# Print pending job for each backend along with current date and time
text = f"\n{dt_string}\t{brisbane.status().pending_jobs}\t{osaka.status().pending_jobs}\t{kyoto.status().pending_jobs}"

# Open target text file
f = open('queue_data.txt', 'a+')

# Write new queue data in text file
f.write(text)

# Close the file
f.close()