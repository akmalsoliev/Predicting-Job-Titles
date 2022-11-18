import sweetviz as sv

def eda_report(df):
    report = sv.analyze(df)
    report.show_html(
            filepath="report/jobs_report.html",
            open_browser=True,
            layout="vertical",
            scale=1.
            )
