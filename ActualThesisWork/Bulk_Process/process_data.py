"""
Script to start processing run ids

Machine midway3 -> Dir scratch -> No Quota
- corresponding mount device -> midway3_perf total capacity 355TB -> Used rn 286TB

caslake device seems to be available for being clogged by these jobs

source /cvmfs/xenon.opensciencegrid.org/releases/nT/el7.2024.10.4/setup.sh

Doing first 20 of each run id index
"""
import os


os.environ["SCRATCH_DIR"] = '/scratch/midway3/fsemler'
os.environ["SCRATCH"] = os.environ["SCRATCH_DIR"]


from utilix.batchq import submit_job

dry_run = False

# Make python script for processing
python_script = """
import straxen
import cutax 
import sys
import shutil
import h5py
import os

run_id = sys.argv[1]

save_path = "/scratch/midway3/fsemler/processed"

in_dir  = "/scratch/midway3/fsemler/in_dat"
out_dir = "/scratch/midway3/fsemler/intermediates"

output_folder = out_dir
_raw_paths = in_dir
_processed_paths = [in_dir, out_dir]

### Online Modified for local
st = cutax.contexts.xenonnt_online(     we_are_the_daq              = False,
                                        output_folder               = output_folder,
                                        include_rucio_remote        = True,
                                        include_online_monitor      = False,
                                        include_rucio_local         = False,
                                        download_heavy              = True,
                                        _auto_append_rucio_local    = True,
                                        _raw_paths                  = _raw_paths,
                                        _processed_paths            = _processed_paths,)

pat = st.get_array(run_id, targets='event_area_per_channel')
ei = st.get_array(run_id, targets='event_info')

with h5py.File(os.path.join(save_path, run_id + '.hdf5'), 'w') as f:
    f.create_dataset('event_area_per_channel', data=pat)
    f.create_dataset('event_info', data=ei)

# Remove all files no longer required
# TODO This does not work -> Whatever fix later
shutil.rmtree('{}/*{}*'.format(in_dir, run_id))
shutil.rmtree('{}/*{}*'.format(out_dir, run_id))
"""

# Write python script to file 
with open('/scratch/midway3/fsemler/process.py', "w") as file:
    file.write(python_script)


save_path = "/scratch/midway3/fsemler/processed"
in_dir  = "/scratch/midway3/fsemler/in_dat"
out_dir = "/scratch/midway3/fsemler/intermediates"

for i in (save_path, in_dir, out_dir):
    if not os.path.isdir(i):
        os.mkdir(i)

# Load run_ids 
with open('/home/fsemler/run_ids.txt', "r") as file:
    run_ids =file.readlines()

run_ids = [i.strip('\n') for i in run_ids]

# And run script
for run_id in range(len(run_ids)):
    print
    jobstring = """
    python3 /scratch/midway3/fsemler/process.py {}
    """.format(run_id)

    # Submit batch Job
    submit_job(jobstring,
               log = '{}.log'.format(run_id),
               partition = 'caslake',
               qos = 'caslake', 
               jobname = 'Process_{}'.format(run_id),
               dry_run = dry_run,
               mem_per_cpu = 120 * 1024, # Memory required for processing
               container = 'xenonnt-el9.2024.10.4.simg',
               verbose = True,
               )