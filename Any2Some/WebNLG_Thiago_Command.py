import os

def train_command(currentpath, category, addition, testpart):
    if addition == 'None':
        os.system('python3 train.py --tokenizer t5-large --model t5-large --src_train WebNLG/' + category + '/trainsrc.txt --trg_train WebNLG/' + category + '/traintrg.txt --src_dev WebNLG/' + category + '/devsrc.txt --trg_dev WebNLG/' + category + '/devtrg.txt --epochs 16 --learning_rate 1e-5 --batch_size 2 --early_stop 5 --max_length 180 --write_path ' + currentpath + '/SavedModels/WebNLG/' + category + '/base --language english --verbose --batch_status 16 --cuda')
    else:
        os.system(
            'python3 train.py --tokenizer t5-large --model t5-large --src_train WebNLG/' + category + '/' + addition + '_train' + testpart + 'src.txt --trg_train WebNLG/' + category + '/' + addition + '_train' + testpart + 'trg.txt --src_dev WebNLG/' + category + '/devsrc.txt --trg_dev WebNLG/' + category + '/devtrg.txt --epochs 16 --learning_rate 1e-5 --batch_size 2 --early_stop 5 --max_length 180 --write_path ' + currentpath + '/SavedModels/WebNLG/' + category + '/' + addition + '/' + testpart + ' --language english --verbose --batch_status 16 --cuda')

def evaluate_command(currentpath, category, devtest, addition, testpart):
    if addition == 'None':
        os.system('python3 evaluate.py --tokenizer t5-large --model ' + currentpath + '/SavedModels/WebNLG/' + category + '/base/model --src_test WebNLG/' + category + '/' + devtest + 'src.txt --trg_test WebNLG/' + category + '/' + devtest + 'trg.txt --batch_size 8 --max_length 180 --write_dir ' + currentpath + '/results/WebNLG/' + category + '/base/' + devtest + ' --language english --verbose --batch_status 16 --cuda')
    else:
        os.system(
            'python3 evaluate.py --tokenizer t5-large --model ' + currentpath + '/SavedModels/WebNLG/' + category + '/' + addition + '/' + testpart + '/model --src_test WebNLG/' + category + '/' + devtest + 'src.txt --trg_test WebNLG/' + category + '/' + devtest + 'trg.txt --batch_size 8 --max_length 180 --write_dir ' + currentpath + '/results/WebNLG/' + category + '/' + addition + '/' + testpart + '/' + devtest + ' --language english --verbose --batch_status 16 --cuda')
currentpath = os.getcwd()
subdirlist = os.listdir(currentpath + '/WebNLG')

for category in subdirlist:
    if os.path.isfile(currentpath + '/WebNLG/' + category + '/trainsrc.txt'):
        additionlist = ['None', 'dat_aug', 'sem_par']
        for addition in additionlist:
            if addition == 'None':
                os.makedirs(currentpath + '/SavedModels/WebNLG/' + category + '/base', exist_ok=True)
                if not os.path.isfile(currentpath + '/SavedModels/WebNLG/' + category + '/base/model/config.json'):
                    train_command(currentpath, category, addition, '')
                devtestlist = ['dev', 'test']
                for devtest in devtestlist:
                    os.makedirs(currentpath + '/results/WebNLG/' + category + '/base/' + devtest, exist_ok=True)
                    if not os.path.isfile(currentpath + '/results/WebNLG/' + category + '/base/' + devtest + '/scores.txt'):
                        evaluate_command(currentpath, category, devtest, addition, '')
            else:
                testpartlist = ['125', '250', '500', '1000']
                for testpart in testpartlist:
                    os.makedirs(currentpath + '/SavedModels/WebNLG/' + category + '/' + addition + '/' + testpart, exist_ok=True)
                    if not os.path.isfile(currentpath + '/SavedModels/WebNLG/' + category + '/' + addition + '/' + testpart + '/model/config.json'):
                        train_command(currentpath, category, addition, testpart)
                    devtestlist = ['dev', 'test']
                    for devtest in devtestlist:
                        os.makedirs(currentpath + '/results/WebNLG/' + category + '/' + addition + '/' + testpart + '/' + devtest, exist_ok=True)
                        if not os.path.isfile(currentpath + '/results/WebNLG/' + category + '/' + addition + '/' + testpart + '/' + devtest + '/scores.txt'):
                            evaluate_command(currentpath, category, devtest, addition, testpart)
