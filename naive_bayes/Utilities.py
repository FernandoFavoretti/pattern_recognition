import openpyxl
import pandas as pd
import os

def to_excel(dataframe, name, sheet_name="Sheet1", append=True):
    #trata o nome recebido para garantir que termina com .xlsx
    if ".xls" in name:
        name = ".".join(name.split(".")[:-1])
    name += ".xlsx"
    
    #cria o writer com a engine openpyxl para poder adicionar uma nova planilha ao arquivo
    writer = pd.ExcelWriter(name, engine="openpyxl")
    wb=None
    #trata exceções para ter certeza que o arquivo nao ficara corrompido se algo der errado
    try:
        if append and os.path.isfile(name):
            wb = openpyxl.load_workbook(name)
            if sheet_name in wb.get_sheet_names():
                del wb[sheet_name]
            writer.book = wb

        #salva o dataframe no excel
        dataframe.to_excel(excel_writer=writer, sheet_name=sheet_name)
    finally:
        #fecha o arquivo
        writer.close()
        if wb:
            wb.close()