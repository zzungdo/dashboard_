from dash import Input, Output, State, ctx, dcc, html
import dash
import numpy as np
from utils import *

# *한글,특수문자 경로 읽는 함수
def imread_unicode(path):
    try:
        with open(path, 'rb') as f:
            data = np.frombuffer(f.read(), np.uint8)
        return cv2.imdecode(data, cv2.IMREAD_COLOR)
    except Exception as e:
        print(f"[ERROR] 이미지 로딩 실패: {path}\n{e}")
        return None

def register_callbacks(app):

    @app.callback(
        Output('image-display', 'figure'), # *결과를 표시할 대상
        Input('image-dropdown', 'value'), # *선택된 이미지 파일명
        Input('class-visibility-filter', 'value'), # *표시할 클래스 목록
        Input('main-tabs', 'value'), # *현재 선택된 탭
        Input('duplicate-table', 'active_cell'), # *중복 검사 테이블의 선택 셀
        State('duplicate-table', 'data'), # *중복 검사 결과 데이블 전체 데이터
        State('duplicate-table', 'page_current'),# *현재 테이블 페이지 
        State('duplicate-table', 'page_size'), # * 테이블 페이지당 행 수
        prevent_initial_call=True
    )
    def update_image_figure(selected_image, visible_classes, current_tab,
                            active_cell, dup_table_data, page_current, page_size):
        global selected_bbox_index
        # *클래스 통계 탭에서 이미지 표시 하지않음
        if current_tab == 'tab-stats':
            return dash.no_update
        
        # *이미지가 선택되지않았으면 텍스트 출력
        if selected_image is None:
            fig = px.imshow(np.ones((10, 10, 3)) * 255)  # 흰 배경
            fig.update_layout(
                annotations=[{
                    'text': "이미지를 선택 해주세요.",
                    'xref': 'paper', 'yref': 'paper',
                    'x': 0.5, 'y': 0.5,
                    'showarrow': False,
                    'font': {'size': 20, 'color': 'gray'},
                    'xref': 'paper', 'yref': 'paper',
                    'xanchor': 'center', 'yanchor': 'middle'
                }],
                xaxis=dict(visible=False),
                yaxis=dict(visible=False),
                margin=dict(l=0, r=0, t=0, b=0)
            )
            return fig

        # *이미지 로딩
        image_path = os.path.join(IMG_FOLDER, selected_image)
        image = imread_unicode(image_path)
        if image is None:
            return px.imshow(np.zeros((100,100,3)))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        h, w = image.shape[:2]
        fig = px.imshow(image)

        # *바운딩박스 로딩
        boxes = df_all[df_all['filename'] == selected_image].copy()
        is_dup_tab = (current_tab == 'tab-duplicate')

        # *산점도 탭에서 바운딩박스 선택된 경우
        if current_tab == 'tab-analysis' and selected_bbox_index is not None:
            boxes = df_all.loc[[selected_bbox_index]]

        # *중복 검사 탭에서 테이블 클릭한 경우
        elif is_dup_tab and active_cell and dup_table_data:
            idx = page_current * page_size + active_cell['row']
            df_img = df_all[df_all['filename'] == selected_image].reset_index(drop=True)
            a, b = int(dup_table_data[idx]['line_a']) - 1, int(dup_table_data[idx]['line_b']) - 1
            boxes = df_img.iloc[[a, b]]

        else:
            boxes = boxes[boxes['class'].isin(visible_classes)] if visible_classes else boxes

        # *바운딩박스 시각화를 위한 리스트 구성
        shapes, annotations = [], []
        for draw_idx, (_, row) in enumerate(boxes.iterrows()):
            # *중복 검사 탭일 경우
            if is_dup_tab:
                if draw_idx == 0:
                    color_rgb = (255, 0, 0)
                elif draw_idx == 1:
                    color_rgb = (0, 255, 0)
                else:
                    color_rgb = (255, 255, 0) 
            # *이외는 고유 지정 색상
            else:
                color_rgb = get_class_color(row['class'])
            # *색상 문자열 - RGBA 형식
            color_str = f"rgba({color_rgb[0]},{color_rgb[1]},{color_rgb[2]},1.0)"
            # *bbox 좌표
            x0, y0, x1, y1 = row['xmin'], row['ymin'], row['xmax'], row['ymax']
            # *bbox 그리기위한 정보
            shapes.append({
                'type': 'rect', 'x0': x0, 'x1': x1, 'y0': y0, 'y1': y1,
                'xref': 'x', 'yref': 'y', 'line': {'color': color_str, 'width': 2}
            })

            # *class name 어디에 적어둘건지
            if is_dup_tab:
                if draw_idx == 0:
                    ann_x, ann_y, anchor = x0, y0 - 8, 'left'
                else:
                    ann_x, ann_y, anchor = x0 + 15, y1 + 10, 'center'
            else:
                ann_x, ann_y, anchor = x0, y0 - 8, 'left'

            annotations.append({
                'x': ann_x, 'y': ann_y, 'text': row['class'], 'showarrow': False,
                'font': {'size': 25, 'color': color_str}, 'xanchor': anchor
            })

        fig.update_layout(
            shapes=shapes, annotations=annotations, dragmode='zoom',
            margin={'l':0,'r':0,'t':30,'b':0},
            xaxis={'scaleanchor':'y','constrain':'domain','range':[0,w]},
            yaxis={'autorange':'reversed','range':[h,0]},
            clickmode='event+select',
            autosize=True
        )
        return fig

    @app.callback(
        Output('center-scatter', 'figure'), # *산점도 그래프
        Input('main-tabs', 'value') # *현재 선택된 탭
    )
    def update_center_scatter(tab):
        # *산점도 탭이 아니면 리턴
        if tab != 'tab-analysis':
            return dash.no_update

        # *산점도 생성    
        fig = px.scatter(
            df_all,
            x='bbox_width',
            y='bbox_height',
            color='class',
            hover_data=['filename', 'class'], # *마우스 오버
            title='바운딩박스 너비 vs 높이 산점도'
        )
        # *산점도 포인트 크기
        fig.update_traces(marker=dict(size=8), selector=dict(mode='markers'))

        # *여백,
        fig.update_layout(margin=dict(l=20, r=20, t=50, b=20), height=600)

        return fig

    @app.callback(
        Output('heatmap', 'figure'),
        Input('heatmap-class-filter', 'value'),
        Input('heatmap-grid-size', 'value')
    )
    def update_heatmap(selected_class, grid_size):
        # 클래스 필터링
        df = df_all if selected_class == 'All' else df_all[df_all['class'] == selected_class]
        df = df.dropna(subset=['xmin_norm', 'xmax_norm', 'ymin_norm', 'ymax_norm'])

        # 히트맵 데이터 계산
        data = compute_heatmap_data(df, grid_size)

        if data.empty:
            return px.imshow(np.zeros((10, 10)), title="히트맵 데이터 없음")

        # 퍼센트 배열 만들기 (히트맵 형태로 reshape 필요)
        total = data['box_count'].sum()
        data['percent'] = (data['box_count'] / total * 100).round(2)
        percent_matrix = np.full((grid_size, grid_size), "", dtype=object)
        for _, row in data.iterrows():
            x, y = int(row['grid_x']), int(row['grid_y'])
            percent_matrix[y, x] = f"{row['percent']}%"

        # box_count 배열 만들기 (히트맵용)
        heatmap_matrix = np.zeros((grid_size, grid_size))
        for _, row in data.iterrows():
            x, y = int(row['grid_x']), int(row['grid_y'])
            heatmap_matrix[y, x] = row['box_count']

        # 히트맵 생성
        import plotly.graph_objects as go
        fig = go.Figure(data=go.Heatmap(
            z=heatmap_matrix,
            text=percent_matrix,
            texttemplate="%{text}",
            textfont={"size": 12, "color": "black"},
            colorscale="RdBu",
            reversescale=True,
            colorbar=dict(title="box_count"),
            customdata=percent_matrix,
            hovertemplate=" %{z} bboxes<br> %{customdata}% of total<extra></extra>",
        ))

        fig.update_layout(
            title=f"히트맵 (바운딩박스 영역 기반): {selected_class}",
            margin=dict(l=40, r=40, t=50, b=40),
            height=600,
        )
        
        fig.update_xaxes(
            title="grid_x",
            range=[-0.5, grid_size - 0.5],
            constrain="domain",
            tickmode="linear",
            dtick=1,
        )
        fig.update_yaxes(
            title="grid_y",
            range=[grid_size - 0.5, -0.5],
            constrain="domain",
            tickmode="linear",
            dtick=1,
        )
        return fig

    @app.callback(
        Output('thumbnail-gallery', 'children'), # *이미지 썸네일 목록
        Input('heatmap', 'clickData'), # *히트맵 클릭 좌표
        Input('main-tabs', 'value'), # *현재 탭
        State('heatmap-class-filter', 'value'), # *클래스 필터
        State('heatmap-grid-size', 'value') # *히트맵 해상도
    )
    def show_thumbnails(clickData, current_tab, sel_class, grid_size):
        # *히트맵 탭 아니거나, 클릭 안한경우
        if current_tab != 'tab-heatmap' or not clickData:
            return []
        # *클릭된 셀의 좌표
        gx = int(clickData['points'][0]['x'])
        gy = int(clickData['points'][0]['y'])
        
        # *클래스 필터링
        df = df_all.copy() if sel_class == 'All' else df_all[df_all['class'] == sel_class]

        # *바운딩박스 중심좌표로 계산
        x0 = gx / grid_size
        x1 = (gx + 1) / grid_size
        y0 = gy / grid_size
        y1 = (gy + 1) / grid_size

        matched = df[
            (df['xmax_norm'] >= x0) & (df['xmin_norm'] <= x1) &
            (df['ymax_norm'] >= y0) & (df['ymin_norm'] <= y1)
        ]

        return [
            html.Img(
                src=get_image_thumbnail(fname),
                id={'type': 'thumb', 'index': fname},
                style={'cursor': 'pointer'}
            )
            for fname in matched['filename'].unique()]

    @app.callback(
        Output('bbox-info', 'children'), # *bbox 정보 리스트
        Input('image-display', 'clickData'), # *클릭 정보
        State('image-dropdown', 'value'), # *선택된 이미지 파일명
        prevent_initial_call=True
    )
    def on_bbox_click(clickData, filename):
        # *클릭 데이터 없거나 형식 잘못된 경우
        if not clickData or 'points' not in clickData:
            return []
        # *클릭한 위치의 좌표
        x, y = clickData['points'][0]['x'], clickData['points'][0]['y']

        # *클릭 위치의 bbox 박스 찾기
        match = df_all[
            (df_all['filename'] == filename) &
            (df_all['xmin'] <= x) & (df_all['xmax'] >= x) &
            (df_all['ymin'] <= y) & (df_all['ymax'] >= y)
        ]
        if match.empty:
            return [html.Div('선택된 바운딩박스 없음', style={'color': 'gray'})]
        return [
            html.Div(f"클래스: {r['class']} / 좌표: ({r['xmin']},{r['ymin']},{r['xmax']},{r['ymax']})")
            for _, r in match.iterrows()
        ]

    @app.callback(
        Output('image-dropdown', 'value', allow_duplicate=True), # *드롭다운 값 갱신
        Input('summary-table', 'active_cell'), # *선택된 테이블 셀
        State('summary-table', 'derived_virtual_data'), # *현재 페이지의 데이터
        State('summary-table', 'page_current'), # *현재 페이지 번호
        State('summary-table', 'page_size'), # *페이지당 행 수
        prevent_initial_call=True
    )
    def update_image_from_summary(active_cell, derived_data, page_current, page_size):
        if active_cell and derived_data:

            # *현재 페이지 내 index
            page_index = active_cell['row']  

            # *전체 데이터 index
            global_index = page_current * page_size + page_index

            if 0 <= global_index < len(derived_data):
                return derived_data[global_index]['filename']
        return dash.no_update

    @app.callback(
        Output('image-dropdown', 'value', allow_duplicate=True), # * 드롭다운 값 갱신
        Input('center-scatter', 'clickData'), # *산점도 클릭
        Input({'type': 'thumb', 'index': dash.ALL}, 'n_clicks'), # *썸네일 이미지 클릭
        State({'type': 'thumb', 'index': dash.ALL}, 'id'), # *썸네일 ID
        prevent_initial_call=True
    )
    def update_image_dropdown(scatter_click, n_clicks, ids):
        global selected_bbox_index
        # *산점도 클릭한 경우
        if ctx.triggered_id == 'center-scatter' and scatter_click:
            fn = scatter_click['points'][0]['customdata'][0]
            x, y = scatter_click['points'][0]['x'], scatter_click['points'][0]['y']
            match = df_all[(df_all['filename'] == fn) & (df_all['bbox_width'] == x) & (df_all['bbox_height'] == y)]
            if not match.empty:
                selected_bbox_index = match.index[0]
            return fn
        # *썸네일 클릭한 경우
        if isinstance(ctx.triggered_id, dict):
            selected_bbox_index = None
            return ctx.triggered_id['index']
        return dash.no_update

    @app.callback(
        Output('duplicate-table', 'data'), # * 중복 검사 결과 테이블 데이터
        Input('btn-duplicate-check', 'n_clicks'), # *검사 시작 버튼 클릭
        State('dup-threshold', 'value'), # *좌표 차이 임계값
        State('dup-type-filter', 'value'), # *비교 타입
        prevent_initial_call=True
    )
    def check_duplicate_boxes(n_clicks, threshold, type_filters):
        global duplicate_box_dict
        duplicate_box_dict = {}
        rows = []
        
        # *중복 바운딩박스 검사
        for fn in sorted(df_all['filename'].unique()):
            df_img = df_all[df_all['filename'] == fn]
            indices = df_img.index.tolist()
            lines = df_img[['class', 'xmin', 'ymin', 'xmax', 'ymax']].values.tolist()

            used_indices = set()
            checked_pairs = set()

            for i in range(len(lines)):
                if indices[i] in used_indices:
                    continue
                for j in range(i + 1, len(lines)):
                    if indices[j] in used_indices:
                        continue
                    pair = (indices[i], indices[j])
                    if pair in checked_pairs or (pair[1], pair[0]) in checked_pairs:
                        continue

                    cls1, *b1 = lines[i]
                    cls2, *b2 = lines[j]
                    
                    # *두 좌표 차이가 임계값 이하인경우
                    if all(abs(a - b) <= threshold for a, b in zip(b1, b2)):

                        if cls1 == cls2 and 'same' in type_filters:
                            note = '같은 클래스'
                            rows.append({'filename': fn, 'line_a': (i+1), 'line_b': (j+1), 'class_name': cls1, 'note': note})

                        elif cls1 != cls2 and 'diff' in type_filters:
                            note = '다른 클래스'
                            rows.append({'filename': fn, 'line_a': (i+1), 'line_b': (j+1), 'class_name': f"{cls1} / {cls2}", 'note': note})
                        # *중복 bbox 기록 
                        duplicate_box_dict.setdefault(fn, set()).update([indices[i], indices[j]])
                        used_indices.update([indices[i], indices[j]])
                        checked_pairs.add(pair)
                        break
        # *결과 DataFrame 생성 
        df_res = pd.DataFrame(rows).reset_index(drop=True)
        df_res.insert(0, 'index', df_res.index + 1)
        return df_res.to_dict('records')

    @app.callback(
        Output('image-dropdown', 'value', allow_duplicate=True),
        Input('duplicate-table', 'active_cell'),
        State('duplicate-table', 'data'),
        State('duplicate-table', 'page_current'),
        State('duplicate-table', 'page_size'),
        prevent_initial_call=True
    )
    def click_dup_table(active_cell, table_data, page_current, page_size):
        if active_cell:
            idx = page_current * page_size + active_cell['row']
            return table_data[idx]['filename'] if idx < len(table_data) else dash.no_update
        return dash.no_update

    @app.callback(
        Output('conditional-display', 'style'),
        Input('main-tabs', 'value')
    )
    def toggle_conditional_display(tab):
        # *통계 탭이면 숨김
        if tab == 'tab-stats':
            return {'display': 'none'}
        return {'display': 'block'}

    @app.callback(
    Output('class-filter-wrapper', 'style'),
    Input('main-tabs', 'value')
    )
    def toggle_class_filter(tab):
        # *분석,통계,검사 탭에서 숨김
        if tab in ['tab-analysis', 'tab-stats', 'tab-duplicate']:
            return {'display': 'none'}
        return {'display': 'block'}


    @app.callback(
        Output('image-dropdown', 'options'),
        Input('main-tabs', 'value'),
        Input('duplicate-table', 'data'),
    )
    def update_dropdown_options(current_tab, dup_table_data):
        if current_tab == 'tab-duplicate' and dup_table_data:
            # *중복 검사 결과에 등장한 파일만 추출
            filenames = sorted(set(item['filename'] for item in dup_table_data))
        else:
            # *전체 목록 사용 (default)
            filenames = sorted(df_all['filename'].unique())

        return [{'label': f, 'value': f} for f in filenames]

    @app.callback(
        Output('image-dropdown', 'style'),
        Input('main-tabs', 'value')
    )
    def toggle_dropdown_visibility(tab):
        # *요약 탭에서만 표시
        if tab == 'tab-summary':
            return {'width': '50%'}  # 기존 스타일 유지
        return {'display': 'none'}  # 그 외 탭에서는 숨김

    @app.callback(
        Output('image-label-display', 'children'),
        Input('image-dropdown', 'value'),
        Input('main-tabs', 'value')
    )
    def display_filename_text(filename, tab):
        if tab == 'tab-summary':
            return ''  # 요약 탭에서는 드롭다운을 사용하므로 텍스트 없음
        if filename:
            return f"선택된 이미지: {filename}"
        return "이미지를 선택해주세요."

    @app.callback(
        Output('image-dropdown', 'value', allow_duplicate=True),
        Input('main-tabs', 'value'),
        prevent_initial_call=True
    )
    def reset_dropdown_on_tab_change(tab):
        if tab != 'tab-summary':
            return None  # 요약 탭 외에서는 초기화
        return dash.no_update  # 요약 탭이면 값 유지
