import sweetviz as sv

def eda_report(df, file_path:str):
    report = sv.analyze(df)
    report.show_html(
            filepath=file_path,
            open_browser=True,
            layout="vertical",
            scale=1.
            )
