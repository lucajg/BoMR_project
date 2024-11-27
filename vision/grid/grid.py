from reportlab.pdfgen import canvas

def create_grid_pdf(output_filename, black_tiles, case_length=14, grid_size=(9, 6), page_size=(91.4, 130), line_thickness=1):
    """
    Create a PDF file with a grid of squares.

    :param output_filename: Name of the PDF file to save.
    :param black_tiles: List of (row, column) tuples specifying which tiles are black.
    :param case_length: Length of each square in centimeters.
    :param grid_size: Tuple (rows, cols) specifying the grid dimensions.
    :param page_size: Tuple (width_cm, height_cm) specifying the page dimensions in centimeters.
    :param line_thickness: Thickness of the grid lines in points.
    """
    # Convert dimensions from cm to points (1 cm = 28.3465 points)
    case_length_points = case_length * 28.3465
    page_width_points = page_size[0] * 28.3465
    page_height_points = page_size[1] * 28.3465

    # Adjust grid size to account for line thickness
    grid_width = grid_size[1] * case_length_points + line_thickness
    grid_height = grid_size[0] * case_length_points + line_thickness

    # Ensure the grid fits within the page
    if grid_width > page_width_points or grid_height > page_height_points:
        raise ValueError("The grid is too large to fit on the page with the given parameters.")

    # Initialize canvas with custom page size
    c = canvas.Canvas(output_filename, pagesize=(page_width_points, page_height_points))

    # Calculate grid origin to center it on the page
    x_origin = (page_width_points - grid_width) / 2
    y_origin = (page_height_points - grid_height) / 2

    # Draw the grid
    for row in range(grid_size[0]):
        for col in range(grid_size[1]):
            x = x_origin + col * case_length_points
            y = y_origin + (grid_size[0] - 1 - row) * case_length_points

            # Draw the square outline
            c.setStrokeColor("black")
            c.setLineWidth(line_thickness)
            c.rect(x, y, case_length_points, case_length_points, fill=0)

            # Fill square if it is in black_tiles
            if (row, col) in black_tiles:
                c.setFillColor("black")
                c.rect(x, y, case_length_points, case_length_points, fill=1, stroke=0)

    # Save the PDF
    c.save()

# Define black tiles as a list of (row, column) tuples
black_tiles = [(1, 1), (2, 0), (3, 3), (4, 2), (6, 5), (6, 4), (8, 1)]  # Example tiles

# Create the grid PDF with a custom page size
create_grid_pdf("custom_page_grid_final.pdf", black_tiles, page_size=(91.4, 130))
