import os

def train_command(currentpath, addition, testpart):
    if addition == 'None':
        os.system('python3 train.py --tokenizer t5-large --model t5-large --src_train E2E/trainsrc.txt --trg_train E2E/traintrg.txt --src_dev E2E/devsrc.txt --trg_dev E2E/devtrg.txt --epochs 16 --learning_rate 1e-5 --batch_size 2 --early_stop 5 --max_length 180 --write_path ' + currentpath + '/SavedModels/E2E/base --language english --verbose --batch_status 16 --cuda')
    else:
        os.system(
            'python3 train.py --tokenizer t5-large --model t5-large --src_train E2E/' + addition + '_train' + testpart + 'src.txt --trg_train E2E/' + addition + '_train' + testpart + 'trg.txt --src_dev E2E/devsrc.txt --trg_dev E2E/devtrg.txt --epochs 16 --learning_rate 1e-5 --batch_size 2 --early_stop 5 --max_length 180 --write_path ' + currentpath + '/SavedModels/E2E/' + addition + '/' + testpart + ' --language english --verbose --batch_status 16 --cuda')

def evaluate_command(currentpath, devtest, addition, testpart):
    if addition == 'None':
        os.system('python3 evaluate.py --tokenizer t5-large --model ' + currentpath + '/SavedModels/E2E/base/model --src_test E2E/' + devtest + 'src.txt --trg_test E2E/' + devtest + 'trg.txt --batch_size 8 --max_length 180 --write_dir ' + currentpath + '/results/E2E/base/' + devtest + ' --language english --verbose --batch_status 16 --cuda')
    else:
        os.system(
            'python3 evaluate.py --tokenizer t5-large --model ' + currentpath + '/SavedModels/E2E/' + addition + '/' + testpart + '/model --src_test E2E/' + devtest + 'src.txt --trg_test E2E/' + devtest + 'trg.txt --batch_size 8 --max_length 180 --write_dir ' + currentpath + '/results/E2E/' + addition + '/' + testpart + '/' + devtest + ' --language english --verbose --batch_status 16 --cuda')

currentpath = os.getcwd()
#additionlist = ['None', 'dat_aug_', 'sem_par_']
additionlist = ['dat_aug', 'sem_par']

for addition in additionlist:
    if addition == None:
        os.makedirs(currentpath + '/SavedModels/E2E/base', exist_ok=True)
        train_command(currentpath, addition, '')
        devtestlist = ['dev', 'test']
        for devtest in devtestlist:
            os.makedirs(currentpath + '/results/E2E/base/' + devtest, exist_ok=True)
            evaluate_command(currentpath, devtest, addition, '')
    else:
        testpartlist = ['125', '250', '500', '1000']
        for testpart in testpartlist:
            os.makedirs(currentpath + '/SavedModels/E2E/' + addition + '/' + testpart, exist_ok=True)
            train_command(currentpath, addition, testpart)
            devtestlist = ['dev', 'test']
            for devtest in devtestlist:
                os.makedirs(currentpath + '/results/E2E/' + addition + '/' + testpart + '/' + devtest, exist_ok=True)
                evaluate_command(currentpath, devtest, addition, testpart)
