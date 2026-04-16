import streamlit as st
import requests
import pandas as pd
from decouple import config
import time
import io
from datetime import datetime, timedelta
from unidecode import unidecode
import re

# Configuración inicial
BASE_URL = 'https://canvas.uautonoma.cl/api/v1'
TOKEN = config("TOKEN")
HEADERS = {"Authorization": f"Bearer {TOKEN}"}

def get_students(course_id):
    url = f"{BASE_URL}/courses/{course_id}/enrollments"
    params = {"type[]": "StudentEnrollment", "per_page": 100}
    students = []
    while url:
        response = requests.get(url, headers=HEADERS, params=params)
        if response.status_code == 200:
            students.extend(response.json())
            url = response.links.get('next', {}).get('url')
        else:
            st.error(f"Error {response.status_code}: No se pudo obtener la lista de estudiantes del curso {course_id}.")
            break
    return students

def check_last_activity(student):
    last_activity = student.get("last_activity_at")
    return '✔️' if last_activity else '❌'

def get_course_info(course_id):
    response = requests.get(f"{BASE_URL}/courses/{course_id}", headers=HEADERS)
    if response.status_code == 200:
        return response.json()
    else:
        st.error(f"Error {response.status_code}: No se pudo obtener la información del curso {course_id}.")
        return None

def get_subaccount_info(sub_account_id):
    response = requests.get(f"{BASE_URL}/accounts/{sub_account_id}", headers=HEADERS)
    if response.status_code == 200:
        return response.json()
    else:
        st.error(f"Error {response.status_code}: No se pudo obtener la información de la subcuenta {sub_account_id}.")
        return None

def get_assignments(course_id):
    assignments = []
    url = f"{BASE_URL}/courses/{course_id}/assignments"
    while url:
        response = requests.get(url, headers=HEADERS)
        if response.status_code == 200:
            data = response.json()
            assignments.extend(data)
            if 'next' in response.links:
                url = response.links['next']['url']
            else:
                url = None
        else:
            st.error(f"Error {response.status_code} al obtener tareas del curso {course_id}: {response.text}")
            return []
    return assignments

def get_submissions(course_id, assignment_id):
    submissions = []
    url = f"{BASE_URL}/courses/{course_id}/assignments/{assignment_id}/submissions?per_page=100"
    while url:
        response = requests.get(url, headers=HEADERS)
        if response.status_code == 200:
            data = response.json()
            submissions.extend(data)
            if 'next' in response.links:
                url = response.links['next']['url']
            else:
                url = None
        else:
            st.error(f"Error {response.status_code} al obtener entregas de la tarea {assignment_id}: {response.text}")
            return []
    return submissions

st.set_page_config(page_title="Participeitor 👌", page_icon="👌", layout="wide")

def main():
    st.title("Analizador y Generador de Reportes de participación") 
    st.write("Con esta app podrás encontrar rápidamente qué estudiantes participaron y cuáles no en un curso de Canvas. Puedes además opcionalmente incluir las columnas para las tareas y ver quien entregó y quien no. Solo ingresa uno o mas IDs de un diplomado y espera la magia 🎩, recuerda que el orden en que pones los IDs es el orden en que saldran en el reporte.")

    with st.form("my_form"):
        courses_input = st.text_input("Ingrese los IDs de los cursos:", "")
        include_assignments = st.checkbox("Incluir entregas en tareas", value=False)
        ver_participacion = st.form_submit_button("Ver participación")

    if ver_participacion and courses_input:
        cleaned_input = courses_input.replace(',', ' ')
        course_ids = [c.strip() for c in cleaned_input.split() if c.strip().isdigit()]

        if not course_ids:
            st.error("No se han ingresado IDs de curso válidos.")
            return

        start_time = time.time()
        st.session_state['results'] = {}
        diplomado_name = None

        with st.spinner("Obteniendo información de todos los cursos..."):
            for course_id in course_ids:
                students = get_students(course_id)
                if not students:
                    continue

                data = []
                for student in students:
                    participation = check_last_activity(student)
                    created_str = student.get("created_at")
                    created = datetime.strptime(created_str, "%Y-%m-%dT%H:%M:%SZ") if created_str else None
                    activity_str = student.get("last_activity_at")
                    activity = datetime.strptime(activity_str, "%Y-%m-%dT%H:%M:%SZ") if activity_str else None
                    total_activity = student.get("total_activity_time")
                    if total_activity is not None:
                        horas = total_activity // 3600
                        minutos = (total_activity % 3600) // 60
                        segundos = total_activity % 60
                        total_activity_formated = f"{horas:02}:{minutos:02}:{segundos:02}"
                    else:
                        total_activity_formated = "00:00:00" 

                    sortable_name_list = student.get('user', {}).get('sortable_name', '').split(',')
                    if len(sortable_name_list) < 2:
                        sortable_name_list = [sortable_name_list[0] if sortable_name_list else "", ""]

                    rut = student.get('user', {}).get("sis_user_id")
                    user_id = student.get('user', {}).get('id')
                    data.append({
                        "Nombres": sortable_name_list[1].strip() if len(sortable_name_list)>1 else "",
                        "Apellidos": sortable_name_list[0].strip() if len(sortable_name_list)>0 else "",
                        "RUT": f"{rut[:-1]}-{rut[-1]}" if rut and len(rut) > 1 else None,
                        "Correo": student.get('user', {}).get("login_id"),
                        "Matriculado": created.strftime("%d-%m-%Y %H:%M") if created else None,
                        "Ultima actividad": activity.strftime("%d-%m-%Y %H:%M") if activity else "Nunca",
                        "Ha participado": participation,
                        "Actividad total": total_activity_formated,
                        "user_id": user_id
                    })
                
                df = pd.DataFrame(data)

                course_info = get_course_info(course_id)
                if not course_info:
                    continue
                sub_account_info = get_subaccount_info(course_info.get("account_id"))

                if diplomado_name is None and sub_account_info:
                    diplomado_name = sub_account_info.get('name', 'Diplomado')

                # Procesar tareas antes de remover user_id
                if include_assignments:
                    assignments = get_assignments(course_id)
                    filtered_assignments = []
                    for a in assignments:
                        normalized_name = unidecode(a['name'].lower())
                        if 'autoevaluacion' not in normalized_name:
                            filtered_assignments.append(a)

                    for a in filtered_assignments:
                        submissions = get_submissions(course_id, a['id'])
                        delivered = set()
                        for s in submissions:
                            wfs = s.get('workflow_state')
                            grd = s.get('grade')
                            if wfs in ['submitted', 'graded']:
                                if grd is not None:
                                    try:
                                        if float(grd) > 0:
                                            delivered.add(s['user_id'])
                                    except:
                                        delivered.add(s['user_id'])
                                else:
                                    delivered.add(s['user_id'])
                        task_name = a['name']
                        df[task_name] = df['user_id'].apply(lambda uid: "✔️" if uid in delivered else "❌")

                # Remover user_id
                if 'user_id' in df.columns:
                    df = df.drop(columns=['user_id'])

                participantes_count = df[df["Ha participado"] == "✔️"].shape[0]
                no_participantes_count = df[df["Ha participado"] == "❌"].shape[0]

                st.session_state['results'][course_id] = {
                    'df': df,
                    'participantes_count': participantes_count,
                    'no_participantes_count': no_participantes_count,
                    'course_info': course_info,
                    'sub_account_info': sub_account_info
                }

        end_time = time.time()
        tiempo_total = end_time - start_time
        st.session_state['tiempo_total'] = tiempo_total
        st.session_state['include_assignments'] = include_assignments
        st.session_state['diplomado_name'] = diplomado_name if diplomado_name else "Diplomado"

    if 'results' in st.session_state and st.session_state['results']:
        st.write(f" ")
        st.write(f"**Tiempo de obtención de datos:** {st.session_state['tiempo_total']:.2f} segundos")
        st.write("¿Cuánto tiempo te ahorraste 😉?")
        mostrar_no_participantes = st.checkbox("Mostrar solo no participantes", value=False)
        st.divider()

        dfs_to_export = []
        minimal_dfs = []
        courses_tasks_info = []

        for course_id, res in st.session_state['results'].items():
            df = res['df'].copy()
            diplomado = f"{res['sub_account_info'].get('name')} - id: {res['sub_account_info'].get('id')}" if res['sub_account_info'] else "Subcuenta desconocida"
            curso_name = res['course_info'].get('name', f"Curso_{course_id}")
            invalid_chars = r'[\[\]\:\*\?\/\\\']'
            curso_name_clean = re.sub(invalid_chars, '_', curso_name)
            curso_name_clean = curso_name_clean.strip()
            sheet_name = curso_name_clean[:31]

            curso = f"{curso_name_clean} - id: {res['course_info'].get('id')}" if res['course_info'] else f"Curso {course_id}"

            if mostrar_no_participantes:
                df_to_show = df[df["Ha participado"] == "❌"].copy()
            else:
                df_to_show = df.copy()

            df_to_show = df_to_show.fillna("")
            # Ordenar sin cambiar nombres a minúsculas, solo por "Nombres" o "Apellidos"
            df_to_show = df_to_show.sort_values(by=["Apellidos"], key=lambda col: col.apply(lambda x: unidecode(str(x)))).reset_index(drop=True)

            participantes_count = df_to_show[df_to_show["Ha participado"] == "✔️"].shape[0]
            no_participantes_count = df_to_show[df_to_show["Ha participado"] == "❌"].shape[0]

            st.markdown(f'<span style="font-size: 28px;">{diplomado}</span>', unsafe_allow_html=True)
            st.markdown(f'<span style="font-size: 22px;">*{curso}*</span>', unsafe_allow_html=True)
            st.markdown(f"**:green[Si participaron en la plataforma:]** {participantes_count} / **:red[No participaron en la plataforma:]** {no_participantes_count}")
            st.dataframe(df_to_show, use_container_width=True)

            dfs_to_export.append((df_to_show, diplomado, curso, sheet_name))

            if st.session_state['include_assignments']:
                columns_to_remove = ["RUT","Correo","Matriculado","Ultima actividad","Ha participado"]
                cols_final = [c for c in df_to_show.columns if c not in columns_to_remove]

                # Aquí no se tocan Nombres/Apellidos, no normalizamos a minúsculas
                if "Nombres" in cols_final:
                    cols_final.remove("Nombres")
                cols_final.insert(0, "Nombres")

                if "Apellidos" in cols_final:
                    cols_final.remove("Apellidos")
                cols_final.insert(1, "Apellidos")

                base_df = df_to_show[cols_final]
                num_tasks = base_df.shape[1] - 2
                courses_tasks_info.append((curso_name_clean, num_tasks))
                minimal_dfs.append(base_df)

        output = io.BytesIO()
        with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
            # Hojas individuales
            for (df_to_show, diplomado, curso, sheet_name) in dfs_to_export:
                df_to_show.to_excel(writer, index=False, startrow=3, sheet_name=sheet_name)
                workbook = writer.book
                worksheet = writer.sheets[sheet_name]

                title_format = workbook.add_format({'bold': True, 'font_size': 14})
                subtitle_format = workbook.add_format({'italic': True, 'font_size': 12})
                worksheet.write(0, 0, diplomado, title_format)
                worksheet.write(1, 0, curso, subtitle_format)

                center_format = workbook.add_format({'align': 'center'})
                border_format = workbook.add_format({
                    'border': 1,
                    'align': 'center',
                    'valign': 'vcenter'
                })
                header_format = workbook.add_format({
                    'bold': True,
                    'text_wrap': True,
                    'valign': 'top',
                    'border': 1
                })
                check_format = workbook.add_format({
                    'border': 1,
                    'align': 'center',
                    'valign': 'vcenter',
                    'font_color': 'green'
                })
                cross_format = workbook.add_format({
                    'border': 1,
                    'align': 'center',
                    'valign': 'vcenter',
                    'font_color': 'red'
                })

                max_row, max_col = df_to_show.shape
                worksheet.set_column(0, max_col - 1, 20, center_format)
                worksheet.set_column(0, 0, 30) # Nombres
                worksheet.set_column(1, 1, 30) # Apellidos
                worksheet.set_column(3, 3, 40) # Correo más ancho

                for col_num, value in enumerate(df_to_show.columns.values):
                    worksheet.write(3, col_num, value, header_format)

                for row in range(max_row):
                    for col in range(max_col):
                        cell_value = df_to_show.iloc[row, col]
                        if cell_value == '✔️':
                            cell_format = check_format
                        elif cell_value == '❌':
                            cell_format = cross_format
                        else:
                            cell_format = border_format
                        worksheet.write(row + 4, col, cell_value, cell_format)

                worksheet.set_default_row(20)

            # Hoja resumen solo si hay minimal_dfs y include_assignments
            if st.session_state['include_assignments'] and minimal_dfs:
                summary_df = None
                for i, mdf in enumerate(minimal_dfs):
                    # No normalizamos nombres a minúsculas, solo aseguramos no haya NaN
                    mdf = mdf.fillna("")
                    # Hacemos el merge normal
                    col_order = ["Nombres","Apellidos"] + [c for c in mdf.columns if c not in ["Nombres","Apellidos"]]
                    mdf = mdf[col_order]
                    if summary_df is None:
                        summary_df = mdf
                    else:
                        summary_df = pd.merge(summary_df, mdf, on=["Nombres","Apellidos"], how="outer", suffixes=("", "_dup"))
                        col_names = []
                        for c in summary_df.columns:
                            if c.endswith("_dup"):
                                new_c = c.replace("_dup", "")
                                col_names.append(new_c)
                            else:
                                col_names.append(c)
                        summary_df.columns = col_names

                summary_df = summary_df.fillna("")

                # Eliminar posibles filas duplicadas en base a Nombres y Apellidos
                summary_df = summary_df.drop_duplicates(subset=["Nombres","Apellidos"], keep='first')
                # Eliminar filas donde Nombres o Apellidos estén vacíos
                summary_df = summary_df[(summary_df["Nombres"] != "") & (summary_df["Apellidos"] != "")]

                # Ordenar y resetear index
                summary_df = summary_df.sort_values(by=["Apellidos","Nombres"]).reset_index(drop=True)

                task_intervals = []
                current_col = 2
                for (course_name, num_tasks) in courses_tasks_info:
                    if num_tasks > 0:
                        start_col = current_col
                        end_col = current_col + num_tasks - 1
                        task_intervals.append((course_name, start_col, end_col))
                        current_col = end_col + 1

                summary_sheet_name = "Resumen Diplomado"
                summary_df.to_excel(writer, index=False, header=False, startrow=4, sheet_name=summary_sheet_name)
                workbook = writer.book
                worksheet = writer.sheets[summary_sheet_name]

                title_format = workbook.add_format({'bold': True, 'font_size': 14})
                header_format = workbook.add_format({
                    'bold': True,
                    'text_wrap': True,
                    'valign': 'top',
                    'border': 1,
                    'align': 'center'
                })
                border_format = workbook.add_format({
                    'border': 1,
                    'align': 'center',
                    'valign': 'vcenter'
                })
                check_format = workbook.add_format({
                    'border': 1,
                    'align': 'center',
                    'valign': 'vcenter',
                    'font_color': 'green'
                })
                cross_format = workbook.add_format({
                    'border': 1,
                    'align': 'center',
                    'valign': 'vcenter',
                    'font_color': 'red'
                })

                worksheet.write(0, 0, st.session_state['diplomado_name'], title_format)

                max_row, max_col = summary_df.shape

                worksheet.set_column(0, 0, 30) # Nombres
                worksheet.set_column(1, 1, 30) # Apellidos
                worksheet.set_column(2, max_col - 1, 20)

                for (course_name, start_col, end_col) in task_intervals:
                    worksheet.merge_range(2, start_col, 2, end_col, course_name, header_format)

                worksheet.write(3, 0, "Nombres", header_format)
                worksheet.write(3, 1, "Apellidos", header_format)
                for col_num, value in enumerate(summary_df.columns.values):
                    if col_num >= 2:
                        worksheet.write(3, col_num, value, header_format)

                data_start_row = 4
                for row_i in range(max_row):
                    for col_i in range(max_col):
                        cell_value = summary_df.iloc[row_i, col_i]
                        if cell_value == '✔️':
                            cell_format = check_format
                        elif cell_value == '❌':
                            cell_format = cross_format
                        else:
                            cell_format = border_format
                        worksheet.write(data_start_row + row_i, col_i, cell_value, cell_format)

                worksheet.set_default_row(20)

        output.seek(0)
        file_name = f"{st.session_state['diplomado_name'] if 'diplomado_name' in st.session_state else 'Diplomado'}.xlsx"
        st.download_button(
            label="Descargar Reporte de Participación",
            data=output,
            file_name=file_name,
            mime='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet'
        )

    elif ver_participacion and not courses_input:
        st.error("Por favor, ingrese al menos un ID de curso válido antes de ver la participación.")

if __name__ == "__main__":
    main()