from dash import Dash
from layout import create_layout # 레이아웃 구성 함수
from callbacks import register_callbacks # 콜백 등록 함수

app = Dash(__name__)
app.layout = create_layout()
app.title = "Dashboard_v1.0.1"  

register_callbacks(app)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8050, debug=True)