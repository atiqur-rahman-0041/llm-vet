#read the output.json file
import json
import re
import sys

def print_patches(patches):
    # print("================================================")
    for key,value in patches.items():
        print(key)
        print(value)
        print("================")


def create_json(data):
    new_data = []
    new_data_v2 = []
    reversed_new_data = []
    reversed_new_data_v2  = []


    key_list = ['summary', 'description','severity', 'cves', 'cwes' ]
    vulnerable_key_list = ["package", "vulnerable_version_range", "first_patched_version", "vulnerable_version"]

    script_done = -1
     
    for item in data:
        script_done += 1
        new_item = {}
        for key in key_list:
            if key == 'cves':
                new_item['cves'] = {'cve_id': item['cve_id'], 'cve_description': item['cve_description'].replace("\\n", "\n") }
            
            elif key == 'description':
                new_item[key] = modify_description(item[key])

            else:
                #new_item[key] = item[key] if key == 'cwes' else item[key].replace("\\n", "\n")
                new_item[key] = item[key]
        
        if len(item["vulnerabilities"]) > 1:
            item["vulnerabilities"] = deal_with_multiple_vulnerabilities(item["vulnerabilities"])

        for key in vulnerable_key_list:
            new_item[key] = item["vulnerabilities"][0][key]

        # new_item['patches'] = discard_files(item["vulnerabilities"][0])
        retrieved_files = discard_files(item["vulnerabilities"][0])

        files = list(retrieved_files.keys())

        patch_dictionary = {}

        reversed_patch_dictionary = {}

        added_dictionary = {}
        removed_dictionary = {}

        for f in files:
            # print(f)
            patch_data = item["vulnerabilities"][0]['patches'][f]

            # print("passing ",f)

            formatted_patch, added_patches, removed_patches, reversed_formatted_patch  = format_patch(patch_data)
        
            # print(formatted_patch)
            # print(len(formatted_patch), len(added_patches), len(removed_patches),"\n=============")  
            for j, i in enumerate(formatted_patch):
                formatted_patch[j] = f"hunk{j+1}: "+group_context(i)[0]
                reversed_formatted_patch[j] = f"hunk{j+1}: "+group_context(reversed_formatted_patch[j])[0]
                # print(formatted_patch[j])

            patch_dictionary[f] = formatted_patch
            reversed_patch_dictionary[f] = reversed_formatted_patch
            # print(patch_dictionary[f])
            # print('*****************')

            for j, i in enumerate(added_patches):
                added_patches[j] = f"hunk{j+1}: "+group_context(i)[1]
                # print(added_patches):[j])
            added_dictionary[f] = added_patches

            for j, i in enumerate(removed_patches):
                removed_patches[j] = f"hunk{j+1}: "+group_context(i)[1]
                # print(removed_patches):[j])
            removed_dictionary[f] = removed_patches

            # v2_patches = [added_patches, removed_patches]
            
        # print_patches(reversed_patch_dictionary)
        
        new_item['patches'] = patch_dictionary
        new_data.append(new_item)

        reversed_new_item = new_item.copy()
        reversed_new_item['patches'] = reversed_patch_dictionary
        reversed_new_data.append(reversed_new_item)

        new_item_v2 = new_item.copy()
        new_item_v2['patches'] = {'Added': added_dictionary, 'removed': removed_dictionary}
        new_data_v2.append(new_item_v2)

        reversed_new_item_v2 = new_item_v2.copy()
        reversed_new_item_v2['patches'] = {'Added': removed_dictionary, 'removed': added_dictionary}
        reversed_new_data_v2.append(reversed_new_item_v2)


        if script_done % 100 == 0:
            print(f'Script {script_done} completed.')
        # if script_done >= 500:
        #     print(f'Script {script_done} completed.')
        # print_patches(added_dictionary)

    return new_data, reversed_new_data, new_data_v2, reversed_new_data_v2

def deal_with_multiple_vulnerabilities(vulnerabilities):
    new_vulnerabilities = []

    for vulnerability in vulnerabilities:
        if vulnerability['package']['ecosystem'] == 'npm' and len(new_vulnerabilities) == 0:
            
            new_vulnerabilities.append(vulnerability)
        else:
            for i in vulnerability['patches']:
                if i in new_vulnerabilities[0]['patches']:
                    new_vulnerabilities[0]['patches'][i] += vulnerability['patches'][i]
                else:
                    new_vulnerabilities[0]['patches'][i] = vulnerability['patches'][i]

    # print(new_vulnerabilities)
    return new_vulnerabilities
 



def discard_files(item):
    
    exclude_extensions = ['md', 'mdx', 'json', 'pdf', 'xml', 'yaml', 'ipynb', 'txt', 'rst', 'yml', 'css', 'html', 'ignore' ,'lock', 'pem', 'hbs', 'toml', '.liquid', 'env']
    ignore_file_names = ['.github/', 'test', 'requirements','setup', 'Pipfile', 'config', 'docs/', 'version', 'spec', 'CHANGELOG', 'examples', 'license', 'LICENSE', 'License' ]

    
    files_dict = item['patches']
    # print("Before exclusion: ", len(files_dict))
    filtered_files_dict = {
    file_name: content.replace("\\n", "\n") 
    for file_name, content in files_dict.items()
    if not any(file_name.endswith(ext) for ext in exclude_extensions) and not any(fn in file_name for fn in ignore_file_names)
    }
    # print("After exclusion: ", len(filtered_files_dict))
    # print("============")
    return filtered_files_dict

def modify_description(data):
    
    markdown_content = data
    modified_content = re.sub(r'### ([Pp]atches|[Rr]eferences|[Rr]eference|patch)\s*.*?(?=\n#|\Z)', '', markdown_content, flags=re.S)
    
    modified_content = modified_content.strip()
    url_pattern = r'(https?://\S+|www\.\S+)'
    modified_content = re.sub(url_pattern, '[LINK]', modified_content)
    # Regular expression to remove standalone `[LINK]` lines
    modified_content = re.sub(r'^\s*\[LINK\]\s*$', '', modified_content, flags=re.MULTILINE)

    modified_content = re.sub(r'\b[Ff]ixed by\s*\[LINK\]\s*', '', modified_content)
    # Clean up extra newlines left by replacements
    modified_content = re.sub(r'\n+', '\n', modified_content).strip()
    # print(modified_content)
    # print("+++++++++++++++++++++++++++++++")
    return modified_content
    

def format_patch(patch_text):
    # print("Formatting patch text...")
    # # Replace literal "\n" with actual newlines
    # patch_text = patch_text.replace("\\n", "\n")
    # print(patch_text,'\n','-----------------------------------------')
    
    # Define regex patterns for different parts of the diff
    hunk_header_pattern = r'@@ -\d+,\d+ \+\d+,\d+ @@'
    remove_pattern = r'^-(.*)'
    add_pattern = r'^\+(.*)'
    context_pattern = r'^[^+-](.*)'

    # Split patch text by lines
    lines = patch_text.splitlines()
    hunk_count = 1

    new_patch = ""
    formatted_patch = []

    reversed_new_patch = ""
    reversed_formatted_patch = []

    added = ""
    removed = ""
    added_patches = []
    removed_patches = []
    
    for line in lines:
        if re.match(hunk_header_pattern, line):
            # Format hunk header
            line = re.sub(r'@@ -\d+,\d+ \+\d+,\d+ @@', '', line)
            if hunk_count!= 1:
                formatted_patch.append(new_patch)
                new_patch = ""

                reversed_formatted_patch.append(reversed_new_patch)
                reversed_new_patch = ""

                added_patches.append(added)
                removed_patches.append(removed)
                added = ""
                removed = ""

            new_patch = new_patch + "hunk "+str(hunk_count)+"\n"

            reversed_new_patch = reversed_new_patch + "hunk "+str(hunk_count)+"\n"

            added = added + "hunk "+str(hunk_count)+"\n"
            removed = removed + "hunk "+str(hunk_count)+"\n"
            # new_patch.append("\nhunk "+str(hunk_count)+"\n")
            hunk_count += 1
            if line != '':
                new_patch = new_patch + f"Context: {line}\n"

                reversed_new_patch = reversed_new_patch + f"Context: {line}\n"

                added = added + f"Context: {line}\n"
                removed = removed + f"Context: {line}\n"
                # new_patch.append(f"Context: {line}")
        elif re.match(remove_pattern, line):
            # Format removed lines
            line = re.sub(r'^-', '', line)
            if line != '': 
                new_patch = new_patch + f"Removed: {line}\n"

                reversed_new_patch = reversed_new_patch + f"Added: {line}\n"

                removed = removed + f"Removed: {line}\n"
                # new_patch.append(f"Removed: {line}")
        elif re.match(add_pattern, line):
            # Format added lines
            line = re.sub(r'^\+', '', line)
            if line!= '':
                new_patch = new_patch + f"Added: {line}\n"

                reversed_new_patch = reversed_new_patch + f"Removed: {line}\n"

                added = added + f"Added: {line}\n"
                # new_patch.append(f"Added: {line}")
        elif re.match(context_pattern, line):
            # Format context lines (neither added nor removed)
            if line!= '':
                new_patch = new_patch + f"Context: {line}\n"

                reversed_new_patch = reversed_new_patch + f"Context: {line}\n"

                added = added + f"Context: {line}\n"
                removed = removed + f"Context: {line}\n"
                # new_patch.append(f"Context: {line}")
    
    # Join formatted lines back into a single string
    formatted_patch.append(new_patch)

    # Join reversed formatted lines back into a single string
    reversed_formatted_patch.append(reversed_new_patch)


    # Add added and removed patches to the list
    added_patches.append(added)
    removed_patches.append(removed)

    return formatted_patch, added_patches, removed_patches, reversed_formatted_patch
    # return formatted_patch

def convert_single_to_multi_line_comments(js_code):
    # Regular expression to match single-line comments
    single_line_comment_pattern = r'//(.*?)(\n|$)'
    
    # Convert single-line comments to multi-line comments
    converted_code = re.sub(single_line_comment_pattern, r'/*\1 */\n', js_code)
    
    return converted_code
    
def minimize_js_keep_multiline_comments(js_code):
    # Step 1: Remove single-line comments
    js_code = re.sub(r'//.*', '', js_code)

    # Step 2: Remove unnecessary whitespace and newlines
    js_code = re.sub(r'\s+', ' ', js_code)  # Replace multiple spaces with a single space
    js_code = re.sub(r'\s*([{};,:()=+\-*/!==><|"])\s*', r'\1', js_code)  # Remove spaces around symbols
    
    # Step 3: Keep only multi-line comments and attach them directly to the relevant code
    multi_line_comments = re.findall(r'/\*.*?\*/', js_code, re.DOTALL)
    for comment in multi_line_comments:
        # Ensure multi-line comments remain formatted with line breaks within the comment
        formatted_comment = re.sub(r'\s*\n\s*', ' ', comment)
        js_code = js_code.replace(comment, formatted_comment)
        
    return js_code


def group_context(text):
    # Initialize list to store grouped results
    grouped_results = []
    current_prefix = None
    current_content = ""

    grouped_results_v2 = []

    # Process each line
    for line in text.strip().splitlines():
        match = re.match(r"(Context|Added|Removed)\s*:\s*(.*)", line.strip())
        if match:
            prefix, content = match.groups()
            content = content.strip()

            # if content is single line comment, convert to multiline comment
            content = convert_single_to_multi_line_comments(content)
            # print("1 prefix: ", prefix, "content: ", content)

            content = minimize_js_keep_multiline_comments(content)
            # print("2 prefix: ", prefix, "content: ", content, "\n")

            # Check if we're still in the same prefix group
            if prefix == current_prefix:
                current_content += content
            else:
                # Append previous content if we switch groups
                if current_content:
                    grouped_results.append(f"{current_prefix}: {current_content}")
                    grouped_results_v2.append(f"{current_content}")
                current_prefix = prefix
                current_content = content

    # Append the final content
    if current_content:
        grouped_results.append(f"{current_prefix}: {current_content}")
        grouped_results_v2.append(f"{current_content}")

    # Print the result
    output = "\n".join(grouped_results)
    output_v2 = "\n".join(grouped_results_v2)
    return output, output_v2


# # Sample patch text with \n included


# Run formatter and print result
# formatted_patch = format_patch(patch_text)
# print(formatted_patch)


def main():
    if len(sys.argv) != 2:
        print("Usage: python3 create_dataset.py <input_json_file>")
        sys.exit(1)
    
    js_file = sys.argv[1]
    new_data = []

    js_file_name = js_file.split(".")[0]
    output_js_file = js_file_name+"_converted.json"
    output_js_file_2 = js_file_name+"_converted_v2.json"
    reverse_output_js_file = js_file_name+"_reversed_converted.json"
    reverse_output_js_file_2 = js_file_name+"_reversed_converted_v2.json"

    with open(js_file) as f:
        data = json.load(f)

        # with open('debug.json', 'w') as f:
        #     json.dump(data[590], f, indent=4)


        new_data, reversed_new_data, new_data_v2, reversed_new_data_v2 = create_json(data)

        # dump a list into a json file
        with open(output_js_file, 'w') as f:
            json.dump(new_data, f, indent=4)

        with open(output_js_file_2, 'w') as f:
            json.dump(reversed_new_data, f, indent=4)
        
        with open(reverse_output_js_file, 'w') as f:
            json.dump(new_data_v2, f, indent=4)

        with open(reverse_output_js_file_2, 'w') as f:
            json.dump(reversed_new_data_v2, f, indent=4)



if __name__ == "__main__":
    main()