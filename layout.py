from dash import html, dcc, dash_table
# !from utils import df_all, df_summary, class_stats, pie_fig

# DASH 기본 구성 요소의 스타일
DEFAULT_STYLES = {
    # *클래스 체크박스 필터 영역
    'filter': {'display': 'block'},
    # *이미지 선택 드롭다운 영역
    'dropdown': {'width': '50%'},
    # *이미지 시각화 영역 스타일
    'image_display': {
        'width': '90%',
        'margin': '0 auto',
        'border': '2px solid gray',
        'marginTop': '5px',
        'aspectRatio': '16 / 9'
    },
    # *바운딩박스 정보 텍스트 영역
    'bbox_info': {'marginTop': '20px'},
    # *히트맵 클릭 시 썸네일 이미지 리스트 영역
    'thumbnail': {
        'display': 'flex',
        'flexWrap': 'wrap',
        'gap': '10px',
        'marginBottom': '20px'
    },
    # *바운딩박스 시각화 제목 스타일
    'title': {'marginTop': '20px'},
    # *바운딩박스 안내 문구 스타일
    'instruction': {'marginTop': '10px'}
}
# * 조건에 따라 숨김 처리할 때 사용하는 스타일
HIDDEN_STYLE = {'display': 'none'}

def create_layout():
    return html.Div([

        html.Div( 
                    id='loading-alert',
                    children="로딩 중입니다. 잠시만 기다려주세요...",
                    style={
                        'display': 'none',
                        'position': 'fixed',
                        'top': '40%',
                        'left': '50%',
                        'transform': 'translate(-50%, -50%)',
                        'backgroundColor': 'rgba(0,0,0,0.85)',
                        'color': 'white',
                        'fontSize': '2em',
                        'padding': '40px',
                        'zIndex': 2000,
                        'borderRadius': '12px'
                    }
                ),
        dcc.Store(id='store-loading'),


        # 상단: 경로 입력 + 적용 버튼 + Store
        html.Div([
            html.Label('GT 폴더:', style={'fontWeight': 'bold'}),
            dcc.Input(id='input-gt-path', type='text', placeholder='./gt', style={'width': '300px', 'height' : '50px'}),
            html.Label('Image 폴더:', style={'fontWeight': 'bold', 'marginLeft': '20px'}),
            dcc.Input(id='input-img-path', type='text', placeholder='./images', style={'width': '300px', 'height' : '50px'}),
            html.Button('적용', id='btn-set-path', style={'width': '50px', 'height' : '50px','marginLeft': '10px', 'color': 'black'}),
        ], style={'position': 'absolute', 'top': '15px', 'right': '30px', 'zIndex': 999, 'background': '#fff', 'padding': '5px', 'borderRadius': '6px'}),
        

        dcc.Loading(
            id="loading-table",
            type="circle",  # "dot" "cube" 등도 가능
            fullscreen=True,  # 화면 전체 오버레이
            color="#d9534f",  # 스피너 색상 (예시)
            children=[
                dash_table.DataTable(id="summary-table"),
                dash_table.DataTable(id="class-summary-table"),
            ]
        ),

        dcc.Store(id='store-paths'),   # 폴더 경로 저장
        dcc.Store(id='store-df-all'), # 전체 GT+Image DataFrame 저장(json)
        dcc.Store(id='store-summary'), # 요약 데이터 저장(json)
        dcc.Store(id='store-class-stats'), # 클래스 통계 저장(json)
        dcc.Store(id='store-pie-fig'),     # 파이차트(fig.to_json()) 저장

        html.H1("Dataset 분석/통계 시각화"),

        dcc.Tabs(id='main-tabs', value='tab-analysis', children=[
            dcc.Tab(label='산점도', value='tab-analysis', children=[
                dcc.Graph(id='center-scatter', figure={})
            ]),
            dcc.Tab(label='히트맵', value='tab-heatmap', children=[
                html.Div([
                    html.Label('클래스 필터:'),
                    dcc.Dropdown(
                        id='heatmap-class-filter',
                        options=[{'label': 'All', 'value': 'All'}],
                        value='All', clearable=False, style={'width': '200px'}
                    ),
                    html.Br(), html.Label('그리드 사이즈:'), 
                    dcc.Input(id='heatmap-grid-size', type='number', value=5, min=1, step=1, style={'width': '100px'})
                ], style={'margin': '10px'}),
                dcc.Graph(id='heatmap')
            ]),
            dcc.Tab(label='클래스 분포 / 통계', value='tab-stats', children=[
                html.Div(
                    dcc.Graph(id='pie-fig', style={'width': '700px', 'height': '700px'}),
                    style={'display': 'flex', 'justifyContent': 'center'}
                ),
                html.Br(), html.H4('클래스별 바운딩박스 통계표'),
                dash_table.DataTable(
                    id='class-summary-table',
                    columns=[
                        {'name': '클래스', 'id': 'class'},
                        {'name': '총 개수', 'id': 'count'},
                        {'name': '클래스 비율(%)', 'id': 'class_ratio_percent'},
                        {'name': '포함 이미지 수', 'id': 'image_count'},
                        {'name': '이미지당 평균 개수', 'id': 'avg_per_image'}
                    ],
                    data=[],
                    style_table={'width': '100%', 'margin': '0 auto'},
                    style_cell={'textAlign': 'center'},
                    style_header={'fontWeight': 'bold'},
                    sort_action='native'
                )
            ]),
            dcc.Tab(label='이미지 요약 / 확인', value='tab-summary', children=[
                dash_table.DataTable(
                    id='summary-table',
                    columns=[
                        {'name': '이미지명', 'id': 'filename'},
                        {'name': '총 바운딩박스 수', 'id': 'bbox_count'},
                        {'name': '클래스 다양성', 'id': 'unique_classes'},
                        {'name': '포함된 클래스', 'id': 'included_classes'}
                    ],
                    data=[],
                    page_current=0, page_size=15,
                    row_selectable='single', sort_action='native', 
                    style_table={'overflowX': 'auto'},
                    style_cell={'textAlign': 'center', 'whiteSpace': 'nowrap'},
                    style_header={'fontWeight': 'bold'},
                    style_data_conditional=[{'if': {'state': 'selected'}, 'backgroundColor': '#D2F3FF', 'border': '1px solid #0074D9'}],
                    style_cell_conditional=[
                        {
                            'if': {'column_id': 'included_classes'},
                            'minWidth': '150px',
                            'maxWidth': '150px',
                            'whiteSpace': 'nowrap',
                            'overflow': 'hidden',
                            'textOverflow': 'ellipsis'
                        }
                    ]
                )
            ]),
            dcc.Tab(label='데이터 검사', value='tab-duplicate', children=[
                html.Div([
                    html.H4('GT 중복 바운딩박스 검사'),
                    html.Label('좌표 임계값(px):'), html.Label('중복 유형 필터:'),
                    dcc.Checklist(
                        id='dup-type-filter',
                        options=[{'label': '같은 클래스', 'value': 'same'}, {'label': '다른 클래스', 'value': 'diff'}],
                        value=['same'], labelStyle={'display': 'inline-block', 'margin-right': '15px'}
                    ),
                    html.Br(),
                    dcc.Input(id='dup-threshold', type='number', value=5, min=0, step=1, style={'width': '100px'}),
                    html.Button('중복 검사 시작', id='btn-duplicate-check', n_clicks=0, style={'marginLeft': '10px'}),
                    html.Br(), html.Br(),
                    dash_table.DataTable(
                        id='duplicate-table',
                        columns=[
                            {'name': 'No', 'id': 'index'},
                            {'name': '이미지명', 'id': 'filename'},
                            {'name': 'Line A', 'id': 'line_a'},
                            {'name': 'Line B', 'id': 'line_b'},
                            {'name': '클래스', 'id': 'class_name'},
                            {'name': '비고', 'id': 'note'}
                        ],
                        data=[], page_size=10, page_current=0, row_selectable='single',
                        style_cell={'textAlign': 'center'}, style_header={'fontWeight': 'bold'},
                        style_table={'overflowX': 'auto'},
                        style_data_conditional=[{'if': {'state': 'selected'}, 'backgroundColor': '#D2F3FF', 'border': '1px solid #0074D9'}]
                    )
                ], style={'padding': '20px'})
            ])
        ]),

        html.Hr(),

        html.Div(id='conditional-display', children=[
            html.Div(id='thumbnail-gallery', style=DEFAULT_STYLES['thumbnail']),
            html.Div(id='bbox-title', children=[html.H3('Bounding Box Display')], style=DEFAULT_STYLES['title']),
            html.Div(id='class-filter-wrapper', children=[
                html.Label('표시할 클래스 선택:'),
                dcc.Checklist(
                    id='class-visibility-filter',
                    options=[], # 콜백으로 업데이트
                    value=[],   # 콜백으로 업데이트
                    inline=True, style={'marginBottom': '20px'}
                )
            ], style=DEFAULT_STYLES['filter']),
            dcc.Dropdown(
                id='image-dropdown',
                options=[], # 콜백으로 업데이트
                value=None,
                style=DEFAULT_STYLES['dropdown']
            ),
            html.Div(id='image-label-display', style={'marginTop': '5px', 'textAlign': 'center', 'fontWeight': 'bold'}),
            html.Br(),
            html.Div([
                html.Div(id='bbox-instruction', children=[
                    html.Label('이미지 바운딩박스 시각화')
                ], style=DEFAULT_STYLES['instruction']),
                dcc.Graph(id='image-display', config={'scrollZoom': False, 'displayModeBar': True, 'displaylogo': False}, style=DEFAULT_STYLES['image_display'])
            ]),
            html.Div(id='bbox-info', style=DEFAULT_STYLES['bbox_info'])
        ])
    ])