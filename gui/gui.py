import os
import sys

sys.path.append('.')
sys.path.append('..')

import pandas as pd
import PySimpleGUI as sg
from src.read.read_xlsx import readDataFromExcel
from src.read.read_csv import readDataFromCSV
from src.eval_indexes.eval_indexes import calc_indexes


# Button Layout
Input_btn = [
    [sg.Text('読み取り対象のファイルとページを選択してください')],
    [sg.Text('ファイル', size=(8, 1)),
     sg.Input(size=(30, 1), disabled=True, key='-FilePath-'),
     sg.FileBrowse(button_text='ファイルを選択', key='-FileInputBtn_Active-')],
    [sg.Text('ページ', size=(8, 1)),
     sg.InputText(default_text='1', size=(10, 1), key='-SheetNum-'),
     sg.Button('読み取り', key='-FileRead-')],
    [sg.Text('ソートする列の\n名前を入力してください', size=(20, 2)),
     sg.InputText(default_text='', size=(10, 1), key='-SortLabel-'),
     sg.Button(button_text='ソート', key='-DataSort-'),
     sg.Button(button_text='指標計算', key='-CalcIndexes-')],
]

table_size = (8, 1)

out_indexes = [
    [sg.Text('', size=(7, 1), pad=(0, 0)),
     sg.Text('micro', size=table_size, pad=(0, 0)),
     sg.Text('macro', size=table_size, pad=(0, 0))
     ],
    [sg.Text('Precision', size=(7, 1), pad=(0, 0)),
     sg.Text(text='', size=table_size, background_color='gray63',
             pad=(0, 0), key='-OutMicroPrecision-'),
     sg.Text(text='', size=table_size, background_color='gray63',
             pad=(0, 0), key='-OutMacroPrecision-')
     ],
    [sg.Text('Recall', size=(7, 1), pad=(0, 0)),
     sg.Text(text='', size=table_size, background_color='gray63',
             pad=(0, 0), key='-OutMicroRecall-'),
     sg.Text(text='', size=table_size, background_color='gray63',
             pad=(0, 0), key='-OutMacroRecall-'),
     ],
    [sg.Text('F値', size=(7, 1), pad=(0, 0)),
     sg.Text(text='', size=table_size, background_color='gray63',
             pad=(0, 0), key='-OutMicroF-'),
     sg.Text(text='', size=table_size, background_color='gray63',
             pad=(0, 0), key='-OutMacroF-'),
     ]
]

# GUI Layout
layout = [
    [sg.Col(Input_btn, background_color='gray63')],
    [sg.Output(size=(55, 20)),
     sg.Col(out_indexes)],
]

window = sg.Window('Analysis Dialog',
                   layout,
                   location=(800, 400))


#--------#
# 環境変数 #
#--------#
param = {'DataRead': False,
         'Data': None}


while True:
    event, values = window.read(timeout=20, timeout_key='-timeout-')

    if event == None or event == sg.WIN_CLOSED:
        break

    elif event != '__TIMEOUT__':
        f = values['-FilePath-']  # CSVファイルパスを取得

        #-------------#
        # データ読み込み #
        #-------------#
        if event == '-FileRead-':
            # ファイルの拡張子が '.xlsx' の場合
            if os.path.splitext(f)[1] == '.xlsx':
                df = readDataFromExcel(
                    f, target_page=int(values['-SheetNum-']))
                print('Excelファイルを読み込みました。')

            # ファイルの拡張子が '.csv' の場合
            elif os.path.splitext(f)[1] == '.csv':
                df = readDataFromCSV(f)
                print('CSVファイルを読み込みました。')

            # 適切な拡張子でない場合
            else:
                print('readできない拡張子です。')
                continue

            param['DataRead'] = True
            param['Data'] = df

        # 以下の処理は、データがreadされていないと動作しない
        if param['DataRead']:
            #-----------#
            # データソート #
            #-----------#
            if event == '-DataSort-':
                df = param['Data'].sort_values(values['-SortLabel-'])
                print(df)

            #---------------#
            # 各指標計算を行う #
            #---------------#
            if event == '-CalcIndexes-':
                indexes = calc_indexes(param['Data'])

                print(indexes)
                # GUI上に表示
                window['-OutMicroPrecision-'].update(
                    str(indexes.at['precision', 'micro']))
                window['-OutMacroPrecision-'].update(
                    str(indexes.at['precision', 'macro']))
                window['-OutMicroRecall-'].update(
                    str(indexes.at['recall', 'micro']))
                window['-OutMacroRecall-'].update(
                    str(indexes.at['recall', 'macro']))
                window['-OutMicroF-'].update(
                    str(indexes.at['f', 'micro']))
                window['-OutMacroF-'].update(
                    str(indexes.at['f', 'macro']))

                # 指標を保存
                dir_path = os.path.split(f)[0]
                dir_path = dir_path + os.sep + 'indexes.csv'
                indexes.to_csv(dir_path)
