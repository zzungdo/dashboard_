from dash import html, dcc, dash_table
from dash.dcc import Loading

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
        # *페이지 상단 헤더 + 경로 입력 UI
        html.Div([
            html.H1("Dataset 분석/통계 시각화"),
        # 오른쪽: 경로 입력 UI
            html.Div([
                # GT 경로
                html.Div([
                    html.Label(
                        "GT : ",
                        style=  {
                                'fontWeight': 'bold',
                                'fontSize': '25px',   
                                'marginRight': '8px'
                                }
                        ),
                    dcc.Input(
                        id='input-gt-path',
                        type='text',
                        placeholder='GT 폴더 경로',
                        style={'width': '300px'}
                    )
                ], style={'marginBottom': '8px'}),

                # Image 경로
                html.Div([
                    html.Label(
                        "IMAGE : ",
                        style=  {
                                'fontWeight': 'bold',
                                'fontSize': '25px',   
                                'marginRight': '8px'
                                }
                        ),
                    dcc.Input(
                        id='input-img-path',
                        type='text',
                        placeholder='Image 폴더 경로',
                        style={'width': '300px'}
                    )
                ], style={'marginBottom': '8px'}),

                # 경로 적용 버튼
                html.Div(
                    html.Button(
                        "RUN",
                        id='btn-apply-path',
                        n_clicks=0,
                        style={
                            'width' : '300px',
                            'padding': '6px 0px'
                            }
                    ),
                    style={'display': 'flex', 'justifyContent': 'flex-end'}
                )
            ],
            style={
                'display': 'flex',
                'flexDirection': 'column',
                'alignItems': 'flex-end',
                'gap': '4px'
            })
        ],
        style={
            'display': 'flex',
            'justifyContent': 'space-between',
            'alignItems': 'center',
            'padding': '10px'
        }),

        # * 입력된 경로를 저장할 Store

        dcc.Loading(
            id='loading-paths',
            type='cube',        # 'default' | 'circle' | 'dot' | 'graph' | 'cube'
            children=dcc.Store(id='store-paths', data={'gt':None,'img':None})
        ),
    
        # *탭 구성
        dcc.Tabs(id='main-tabs', value='tab-analysis', children=[

            # *산점도탭
            dcc.Tab(label='산점도', value='tab-analysis', children=[
                Loading(
                    id='loading-center-scatter',
                    type='cube',
                    children=dcc.Graph(id='center-scatter', figure={})
                )
            ]),

            # *히트맵 탭
            dcc.Tab(label='히트맵', value='tab-heatmap', children=[
                html.Div([
                    html.Label('클래스 필터:'),
                    dcc.Dropdown(
                        id='heatmap-class-filter',
                        options=[{'label': 'All', 'value': 'All'}],  # df_all 참조 제거
                        value='All',                               # 기본값 유지
                        clearable=False, style={'width': '200px'}
                    ),
                    html.Br(), html.Label('그리드 사이즈:'), 
                    dcc.Input(id='heatmap-grid-size', type='number', value=5, min=1, step=1, style={'width': '100px'})
                ], style={'margin': '10px'}),
                
                Loading(
                    id='loading-center-scatter',
                    type='cube',
                    children=dcc.Graph(id='heatmap')
                )
            ]),

            # *클래스 분포 및 통계 탭
            dcc.Tab(label='클래스 분포 / 통계', value='tab-stats', children=[
                # 빈 Pie 차트(콜백으로 채워질 예정)
                html.Div(
                    Loading(
                        id='loading-pie-chart',
                        type='cube',
                        children=dcc.Graph(id='pie-chart', figure={}, style={'width':'700px','height':'700px'})
                    ),
                    style={'display': 'flex', 'justifyContent': 'center'}),
                html.Br(), html.H4('클래스별 바운딩박스 통계표'),
                dash_table.DataTable(
                    id='class-summary-table',
                    columns=[
                        {'name': '클래스',         'id': 'class'},
                        {'name': '총 개수',       'id': 'count'},
                        {'name': '클래스 비율(%)','id': 'class_ratio_percent'},
                        {'name': '포함 이미지 수','id': 'image_count'},
                        {'name': '이미지당 평균 개수','id': 'avg_per_image'}
                    ],
                    data=[],               # 빈 데이터
                    page_size=10,
                    sort_action='native',
                    style_table={'width': '100%', 'margin': '0 auto'},
                    style_cell={'textAlign': 'center'},
                    style_header={'fontWeight': 'bold'}
                )
            ]),
            
            # *이미지별 바운딩박스 요약 / 확인 탭
            dcc.Tab(label='이미지 요약 / 확인', value='tab-summary', children=[
                Loading(
                    id='loading-summary-table',
                    type='cube',
                    children=dash_table.DataTable(
                
                    id='summary-table',
                    columns=[
                        {'name': '이미지명',          'id': 'filename'},
                        {'name': '총 바운딩박스 수',  'id': 'bbox_count'},
                        {'name': '클래스 다양성',    'id': 'unique_classes'},
                        {'name': '포함된 클래스',    'id': 'included_classes'}
                    ],
                    data=[],               # 빈 데이터
                    page_current=0, page_size=15,
                    row_selectable='single', sort_action='native', 
                    style_table={'overflowX': 'auto'},
                    style_cell={'textAlign': 'center','whiteSpace': 'nowrap'},
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
                )
            ]),

            # *데이터 검사 탭
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
                    
                    dcc.Loading(
                        id='loading-duplicate-table',
                        type='cube',
                        children=dash_table.DataTable(
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
                    )
                    ], style={'padding': '20px'})
                ])
            ]),

        # *탭 하단 구분선
        html.Hr(),

        # *탭 아래 표시되는 영역들
        html.Div(id='conditional-display', children=[
            html.Div(id='thumbnail-gallery', style=DEFAULT_STYLES['thumbnail']),
            html.Div(id='bbox-title', children=[html.H3('Bounding Box Display')], style=DEFAULT_STYLES['title']),

            # *클래스 필터 체크박스
            html.Div(id='class-filter-wrapper', children=[
                html.Label('표시할 클래스 선택:'),
            dcc.Checklist(
                id='class-visibility-filter',
                options=[],    # df_all 참조 제거
                value=[],      # 초기값 빈 리스트
                inline=True,
                style={'marginBottom': '20px'}
            )
            ], style=DEFAULT_STYLES['filter']),


            # *이미지 선택 드롭다운 
            dcc.Dropdown(
                id='image-dropdown',
                options=[],    # df_all 참조 제거
                value=None,    # 초기값 None
                style=DEFAULT_STYLES['dropdown']
            ),

            # *선택된 이미지 파일명 출력
            html.Div(id='image-label-display', style={'marginTop': '5px', 'textAlign': 'center', 'fontWeight': 'bold'}),

            html.Br(),

            # *이미지 + 바운딩박스 시각화 영역
            html.Div([
                html.Div(id='bbox-instruction', children=[
                    html.Label('이미지 바운딩박스 시각화')
                ], style=DEFAULT_STYLES['instruction']),
            dcc.Loading(
                id='loading-image-display',
                type='dot',
                children=dcc.Graph(
                    id='image-display',
                    config={'scrollZoom': False, 'displayModeBar': True, 'displaylogo': False},
                    style=DEFAULT_STYLES['image_display'])
                )
            ]),

            # *바운딩박스 클릭 시 정보 출력
            html.Div(id='bbox-info', style=DEFAULT_STYLES['bbox_info'])
        ])
    ])