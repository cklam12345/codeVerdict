import re
import os
from pathlib import Path


def extract_files_from_markdown(markdown_content, output_dir="."):
    """
    Extract files from markdown content with **filename** headers followed by code blocks.
    
    Args:
        markdown_content: String containing the markdown content
        output_dir: Directory where files should be saved (default: current directory)
    """
    # Create output directory if it doesn't exist
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    # Pattern to match **filename** followed by a code block
    pattern = r'\*\*([^\*]+)\*\*\s*```[^\n]*\n(.*?)```'
    
    # Find all matches
    matches = re.findall(pattern, markdown_content, re.DOTALL)
    
    extracted_files = []
    
    for filename, content in matches:
        filename = filename.strip()
        content = content.rstrip('\n')
        
        # Create full path
        filepath = os.path.join(output_dir, filename)
        
        # Create subdirectories if needed
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        # Write file
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(content)
        
        extracted_files.append(filepath)
        print(f"✓ Extracted: {filepath}")
    
    return extracted_files


def main():
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Extract files from markdown with **filename** headers'
    )
    parser.add_argument(
        'input_file',
        help='Path to the markdown file to process'
    )
    parser.add_argument(
        '-o', '--output-dir',
        default='.',
        help='Output directory for extracted files (default: current directory)'
    )
    
    args = parser.parse_args()
    
    # Check if input file exists
    if not os.path.exists(args.input_file):
        print(f"Error: File '{args.input_file}' not found")
        return 1
    
    # Read the markdown file
    try:
        with open(args.input_file, 'r', encoding='utf-8') as f:
            markdown_content = f.read()
    except Exception as e:
        print(f"Error reading file: {e}")
        return 1
    
    # Extract files
    print(f"Processing: {args.input_file}")
    print(f"Output directory: {args.output_dir}\n")
    
    files = extract_files_from_markdown(markdown_content, output_dir=args.output_dir)
    
    if files:
        print(f"\n✓ {len(files)} file(s) extracted successfully!")
    else:
        print("\nNo files found in the markdown.")
    
    return 0


if __name__ == "__main__":
    exit(main())
