import pandas as pd


def get_student_performance(df, student_identifier):
    try:
        search_term = str(student_identifier).strip()
        if not search_term:
            return None
        print(f"🔍 Searching for student: '{search_term}'")
        student_row = None
        if search_term.isdigit():
            print("  → Searching by numeric ID...")
            if 'SR.No' in df.columns:
                sr_matches = df[df['SR.No'].astype(str).str.strip() == search_term]
                if not sr_matches.empty:
                    student_row = sr_matches
                    print(f"  ✅ Found by SR.No: {len(sr_matches)} matches")
            if student_row is None or student_row.empty:
                if 'Roll No' in df.columns:
                    roll_matches = df[df['Roll No'].astype(str).str.strip() == search_term]
                    if not roll_matches.empty:
                        student_row = roll_matches
                        print(f"  ✅ Found by Roll No: {len(roll_matches)} matches")
        if student_row is None or student_row.empty:
            print("  → Searching by name...")
            if 'Name' in df.columns:
                name_matches = df[df['Name'].astype(str).str.contains(search_term, case=False, na=False)]
                if not name_matches.empty:
                    student_row = name_matches
                    print(f"  ✅ Found by Name: {len(name_matches)} matches")
        if student_row is not None and not student_row.empty:
            result = student_row.iloc[0].to_dict()
            print(f"  ✅ Returning student: {result.get('Name', 'Unknown')}")
            return result
        else:
            print(f"  ❌ No student found for: '{search_term}'")
            return None
    except Exception as e:
        print(f"❌ Error in student search: {e}")
        import traceback
        traceback.print_exc()
        return None


