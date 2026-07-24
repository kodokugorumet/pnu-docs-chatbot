package chatbot.parser;

import java.io.IOException;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.List;
import java.util.Locale;
import java.util.regex.Matcher;
import java.util.regex.Pattern;

import kr.dogfoot.hwplib.object.HWPFile;
import kr.dogfoot.hwplib.object.bodytext.Section;
import kr.dogfoot.hwplib.object.bodytext.control.Control;
import kr.dogfoot.hwplib.object.bodytext.control.ControlTable;
import kr.dogfoot.hwplib.object.bodytext.control.ControlType;
import kr.dogfoot.hwplib.object.bodytext.control.table.Cell;
import kr.dogfoot.hwplib.object.bodytext.control.table.Row;
import kr.dogfoot.hwplib.object.bodytext.paragraph.Paragraph;
import kr.dogfoot.hwplib.object.docinfo.Style;
import kr.dogfoot.hwplib.reader.HWPReader;
import kr.dogfoot.hwplib.tool.textextractor.ForControl;
import kr.dogfoot.hwplib.tool.textextractor.ForParagraph;
import kr.dogfoot.hwplib.tool.textextractor.TextExtractMethod;
import kr.dogfoot.hwplib.tool.textextractor.TextExtractOption;
import kr.dogfoot.hwplib.tool.textextractor.paraHead.ParaHeadMaker;
import kr.dogfoot.hwpxlib.object.HWPXFile;
import kr.dogfoot.hwpxlib.reader.HWPXReader;
import kr.dogfoot.hwpxlib.tool.textextractor.TextMarks;
import kr.dogfoot.hwpxlib.tool.textextractor.TextExtractor;

/**
 * HWP/HWPX one-shot worker.
 *
 * <p>The Python orchestrator owns canonical IDs.  This worker emits semantic
 * block candidates and preserves table cells; every block is normalized by
 * the adapter before it enters blocks.jsonl.</p>
 */
public final class HwpParserWorker {
    private static final Pattern HEADING_LEVEL =
            Pattern.compile("(?:제목|개요|heading|title)\\s*([1-6])", Pattern.CASE_INSENSITIVE);
    private static final Pattern LIST_PREFIX =
            Pattern.compile("^(?:[\\u2022\\u25CF\\u25CB\\u25A0\\uF000-\\uF8FF]|\\d+[.)]|[가-힣][.)])\\s*");
    private static final String TABLE_START = "\uE000TABLE_START\uE001";
    private static final String TABLE_END = "\uE000TABLE_END\uE001";
    private static final String ROW_SEPARATOR = "\uE000ROW\uE001";
    private static final String CELL_SEPARATOR = "\uE000CELL\uE001";
    private static final String PARA_SEPARATOR = "\uE000PARA\uE001";
    private static final int MAX_BLOCKS = 250000;
    private static final long MAX_TABLE_CELLS = 1000000L;

    private HwpParserWorker() {
    }

    private static final class Args {
        Path input;
        Path output;
        String profile = "baseline";
    }

    private static final class Block {
        String type;
        String text;
        List<String> sectionPath;
        String tableId;
        Integer row;
        Integer column;

        Block(String type, String text, List<String> sectionPath) {
            this.type = type;
            this.text = "table".equals(type) ? cleanTable(text) : clean(text);
            this.sectionPath = sectionPath == null || sectionPath.isEmpty()
                    ? null : new ArrayList<String>(sectionPath);
        }
    }

    public static void main(String[] argv) throws Exception {
        Args args = parseArgs(argv);
        boolean hwpx = isZip(args.input);
        List<Block> blocks = hwpx ? parseHwpx(args.input) : parseHwp(args.input);
        if (blocks.size() > MAX_BLOCKS) {
            throw new IllegalArgumentException(
                    "worker produced more than " + MAX_BLOCKS + " blocks");
        }
        String library = hwpx ? "hwpxlib-1.0.8" : "hwplib-1.1.10";
        writePayload(args.output, blocks, args.profile, library);
    }

    private static Args parseArgs(String[] argv) {
        Args args = new Args();
        for (int index = 0; index < argv.length; index++) {
            String value = argv[index];
            if ("--input".equals(value) && index + 1 < argv.length) {
                args.input = Paths.get(argv[++index]).toAbsolutePath().normalize();
            } else if ("--output".equals(value) && index + 1 < argv.length) {
                args.output = Paths.get(argv[++index]).toAbsolutePath().normalize();
            } else if ("--profile".equals(value) && index + 1 < argv.length) {
                args.profile = argv[++index];
            } else {
                throw new IllegalArgumentException("Unknown or incomplete argument: " + value);
            }
        }
        if (args.input == null || args.output == null) {
            throw new IllegalArgumentException(
                    "Usage: java -jar hwp-parser-worker.jar --input FILE --output FILE [--profile NAME]");
        }
        return args;
    }

    private static boolean isZip(Path path) throws IOException {
        byte[] header = new byte[4];
        try (java.io.InputStream stream = Files.newInputStream(path)) {
            int count = stream.read(header);
            return count >= 2 && header[0] == 'P' && header[1] == 'K';
        }
    }

    private static List<Block> parseHwp(Path path) throws Exception {
        HWPFile file = HWPReader.fromFile(path.toFile());
        List<Block> blocks = new ArrayList<Block>();
        List<String> sections = new ArrayList<String>();
        ParaHeadMaker paraHeadMaker = new ParaHeadMaker(file);
        int tableIndex = 0;

        for (Section section : file.getBodyText().getSectionList()) {
            paraHeadMaker.startSection(section);
            for (int paragraphIndex = 0;
                 paragraphIndex < section.getParagraphCount();
                 paragraphIndex++) {
                Paragraph paragraph = section.getParagraph(paragraphIndex);
                String styleName = styleName(file, paragraph);
                String text = extractParagraph(paragraph, paraHeadMaker, true);
                if (!text.isEmpty()) {
                    String type = typeFor(styleName, text);
                    if ("heading".equals(type)) {
                        updateSections(sections, headingLevel(styleName), text);
                    }
                    blocks.add(new Block(type, text, sections));
                }

                List<Control> controls = paragraph.getControlList();
                if (controls == null) {
                    controls = Collections.emptyList();
                }
                for (Control control : controls) {
                    if (control.getType() == ControlType.Table) {
                        tableIndex += 1;
                        addHwpTable(
                                blocks,
                                (ControlTable) control,
                                sections,
                                "worker:t" + pad4(tableIndex));
                    } else {
                        String controlText = extractControl(control, paraHeadMaker);
                        if (!controlText.isEmpty()) {
                            blocks.add(new Block("paragraph", controlText, sections));
                        }
                    }
                }
            }
            paraHeadMaker.endSection();
        }
        return blocks;
    }

    private static String styleName(HWPFile file, Paragraph paragraph) {
        int styleIndex = paragraph.getHeader().getStyleId() & 0xffff;
        List<Style> styles = file.getDocInfo().getStyleList();
        if (styleIndex < 0 || styleIndex >= styles.size()) {
            return "";
        }
        Style style = styles.get(styleIndex);
        String name = style.getHangulName();
        if (name == null || name.trim().isEmpty()) {
            name = style.getEnglishName();
        }
        return name == null ? "" : name.trim();
    }

    private static String extractParagraph(
            Paragraph paragraph,
            ParaHeadMaker paraHeadMaker,
            boolean insertParaHead) throws Exception {
        TextExtractOption option = new TextExtractOption(TextExtractMethod.OnlyMainParagraph);
        option.setAppendEndingLF(false);
        option.setWithControlChar(false);
        option.setInsertParaHead(insertParaHead);
        StringBuffer output = new StringBuffer();
        ForParagraph.extract(
                paragraph,
                ForParagraph.ParaStart,
                ForParagraph.ParaEnd,
                false,
                option,
                insertParaHead ? paraHeadMaker : null,
                output);
        return clean(output.toString());
    }

    private static String extractControl(Control control, ParaHeadMaker paraHeadMaker)
            throws Exception {
        TextExtractOption option =
                new TextExtractOption(TextExtractMethod.AppendControlTextAfterParagraphText);
        option.setWithControlChar(false);
        option.setAppendEndingLF(true);
        option.setInsertParaHead(false);
        StringBuffer output = new StringBuffer();
        ForControl.extract(control, option, paraHeadMaker, output);
        return clean(output.toString());
    }

    private static String extractCell(Cell cell) throws Exception {
        List<String> parts = new ArrayList<String>();
        for (int index = 0; index < cell.getParagraphList().getParagraphCount(); index++) {
            String text = extractParagraph(
                    cell.getParagraphList().getParagraph(index),
                    null,
                    false);
            if (!text.isEmpty()) {
                parts.add(text);
            }
        }
        return String.join("\n", parts);
    }

    private static void addHwpTable(
            List<Block> blocks,
            ControlTable table,
            List<String> sections,
            String tableId) throws Exception {
        int rowCount = Math.max(table.getTable().getRowCount(), table.getRowList().size());
        int columnCount = Math.max(1, table.getTable().getColumnCount());
        validateTableSize(rowCount, columnCount);
        String[][] grid = new String[rowCount][columnCount];
        for (String[] row : grid) {
            Arrays.fill(row, "");
        }

        for (Row row : table.getRowList()) {
            for (Cell cell : row.getCellList()) {
                int rowIndex = cell.getListHeader().getRowIndex();
                int columnIndex = cell.getListHeader().getColIndex();
                if (rowIndex >= 0 && rowIndex < rowCount
                        && columnIndex >= 0 && columnIndex < columnCount) {
                    grid[rowIndex][columnIndex] = extractCell(cell);
                }
            }
        }

        addGrid(blocks, grid, sections, tableId);
        if (table.getCaption() != null) {
            List<String> parts = new ArrayList<String>();
            for (int index = 0;
                 index < table.getCaption().getParagraphList().getParagraphCount();
                 index++) {
                String text = extractParagraph(
                        table.getCaption().getParagraphList().getParagraph(index),
                        null,
                        false);
                if (!text.isEmpty()) {
                    parts.add(text);
                }
            }
            if (!parts.isEmpty()) {
                blocks.add(new Block("caption", String.join("\n", parts), sections));
            }
        }
    }

    private static List<Block> parseHwpx(Path path) throws Exception {
        HWPXFile file = HWPXReader.fromFilepath(path.toString());
        TextMarks marks = new TextMarks()
                .lineBreakAnd("\n")
                .paraSeparatorAnd(PARA_SEPARATOR)
                .tableStartAnd(TABLE_START)
                .tableEndAnd(TABLE_END)
                .tableRowSeparatorAnd(ROW_SEPARATOR)
                .tableCellSeparatorAnd(CELL_SEPARATOR);
        String extracted = TextExtractor.extract(
                file,
                kr.dogfoot.hwpxlib.tool.textextractor.TextExtractMethod
                        .AppendControlTextAfterParagraphText,
                true,
                marks);
        return parseMarkedHwpx(extracted);
    }

    private static List<Block> parseMarkedHwpx(String extracted) {
        List<Block> blocks = new ArrayList<Block>();
        int cursor = 0;
        int tableIndex = 0;
        while (cursor < extracted.length()) {
            int tableStart = extracted.indexOf(TABLE_START, cursor);
            if (tableStart < 0) {
                addParagraphs(blocks, extracted.substring(cursor));
                break;
            }
            addParagraphs(blocks, extracted.substring(cursor, tableStart));
            int contentStart = tableStart + TABLE_START.length();
            int tableEnd = findMatchingTableEnd(extracted, contentStart);
            if (tableEnd < 0) {
                addParagraphs(blocks, extracted.substring(contentStart));
                break;
            }
            tableIndex += 1;
            String tableContent = extracted.substring(contentStart, tableEnd);
            List<String> rowValues = splitTopLevel(tableContent, ROW_SEPARATOR);
            List<List<String>> rows = new ArrayList<List<String>>();
            int columns = 1;
            for (String rowValue : rowValues) {
                List<String> cells = splitTopLevel(rowValue, CELL_SEPARATOR);
                columns = Math.max(columns, cells.size());
                rows.add(cells);
            }
            validateTableSize(rows.size(), columns);
            String[][] grid = new String[rows.size()][columns];
            for (int row = 0; row < grid.length; row++) {
                Arrays.fill(grid[row], "");
                for (int column = 0; column < rows.get(row).size(); column++) {
                    grid[row][column] = clean(rows.get(row).get(column));
                }
            }
            addGrid(
                    blocks,
                    grid,
                    Collections.<String>emptyList(),
                    "worker:t" + pad4(tableIndex));
            cursor = tableEnd + TABLE_END.length();
        }
        return blocks;
    }

    private static int findMatchingTableEnd(String value, int contentStart) {
        int depth = 1;
        int cursor = contentStart;
        while (cursor < value.length()) {
            int nextStart = value.indexOf(TABLE_START, cursor);
            int nextEnd = value.indexOf(TABLE_END, cursor);
            if (nextEnd < 0) {
                return -1;
            }
            if (nextStart >= 0 && nextStart < nextEnd) {
                depth += 1;
                cursor = nextStart + TABLE_START.length();
                continue;
            }
            depth -= 1;
            if (depth == 0) {
                return nextEnd;
            }
            cursor = nextEnd + TABLE_END.length();
        }
        return -1;
    }

    private static List<String> splitTopLevel(String value, String delimiter) {
        List<String> values = new ArrayList<String>();
        int depth = 0;
        int start = 0;
        int cursor = 0;
        while (cursor < value.length()) {
            if (value.startsWith(TABLE_START, cursor)) {
                depth += 1;
                cursor += TABLE_START.length();
                continue;
            }
            if (value.startsWith(TABLE_END, cursor)) {
                depth = Math.max(0, depth - 1);
                cursor += TABLE_END.length();
                continue;
            }
            if (depth == 0 && value.startsWith(delimiter, cursor)) {
                values.add(value.substring(start, cursor));
                cursor += delimiter.length();
                start = cursor;
                continue;
            }
            cursor += 1;
        }
        values.add(value.substring(start));
        return values;
    }

    private static void addParagraphs(List<Block> blocks, String value) {
        for (String part : value.split(Pattern.quote(PARA_SEPARATOR), -1)) {
            String text = clean(part);
            if (text.isEmpty()) {
                continue;
            }
            String type = LIST_PREFIX.matcher(text).find() ? "list_item" : "paragraph";
            blocks.add(new Block(type, text, null));
        }
    }

    private static void addGrid(
            List<Block> blocks,
            String[][] grid,
            List<String> sections,
            String tableId) {
        int columns = grid.length == 0 ? 0 : grid[0].length;
        validateTableSize(grid.length, columns);
        if ((long) blocks.size() + 1L + (long) grid.length * columns > MAX_BLOCKS) {
            throw new IllegalArgumentException(
                    "worker would exceed the block limit while expanding a table");
        }
        List<String> lines = new ArrayList<String>();
        for (String[] row : grid) {
            List<String> cells = new ArrayList<String>();
            for (String cell : row) {
                cells.add(clean(cell).replace("\t", " "));
            }
            lines.add(String.join("\t", cells));
        }
        Block parent = new Block("table", String.join("\n", lines), sections);
        parent.tableId = tableId;
        blocks.add(parent);
        for (int row = 0; row < grid.length; row++) {
            for (int column = 0; column < grid[row].length; column++) {
                Block cell = new Block("table_cell", grid[row][column], sections);
                cell.tableId = tableId;
                cell.row = row;
                cell.column = column;
                blocks.add(cell);
            }
        }
    }

    private static void validateTableSize(int rows, int columns) {
        if (rows < 0
                || columns < 0
                || (long) rows * (long) columns > MAX_TABLE_CELLS) {
            throw new IllegalArgumentException(
                    "table exceeds the " + MAX_TABLE_CELLS + " cell limit");
        }
    }

    private static String typeFor(String styleName, String text) {
        String normalized = styleName.toLowerCase(Locale.ROOT);
        if (normalized.contains("제목")
                || normalized.contains("개요")
                || normalized.contains("heading")
                || normalized.contains("title")) {
            return "heading";
        }
        if (normalized.contains("목록")
                || normalized.contains("list")
                || LIST_PREFIX.matcher(text).find()) {
            return "list_item";
        }
        return "paragraph";
    }

    private static int headingLevel(String styleName) {
        Matcher matcher = HEADING_LEVEL.matcher(styleName);
        return matcher.find() ? Integer.parseInt(matcher.group(1)) : 1;
    }

    private static void updateSections(List<String> sections, int level, String text) {
        while (sections.size() >= Math.max(1, level)) {
            sections.remove(sections.size() - 1);
        }
        sections.add(text);
    }

    private static String clean(String value) {
        return clean(value, false);
    }

    private static String cleanTable(String value) {
        return clean(value, true);
    }

    private static String clean(String value, boolean preserveTabs) {
        if (value == null) {
            return "";
        }
        String normalized = value
                .replace(TABLE_START, "")
                .replace(TABLE_END, "")
                .replace(ROW_SEPARATOR, "\n")
                .replace(CELL_SEPARATOR, "\t")
                .replace(PARA_SEPARATOR, "\n")
                .replace('\u0000', ' ')
                .replace("\r\n", "\n")
                .replace('\r', '\n');
        normalized = preserveTabs
                ? normalized
                        .replaceAll(" +", " ")
                        .replaceAll(" *\\t *", "\t")
                : normalized.replaceAll("[\\t ]+", " ");
        return normalized
                .replaceAll("\\n[\\t ]+", "\n")
                .replaceAll("\\n{3,}", "\n\n")
                .trim();
    }

    private static String pad4(int value) {
        return String.format(Locale.ROOT, "%04d", value);
    }

    private static String jsonString(String value) {
        if (value == null) {
            return "null";
        }
        StringBuilder output = new StringBuilder("\"");
        for (int index = 0; index < value.length(); index++) {
            char character = value.charAt(index);
            switch (character) {
                case '"':
                    output.append("\\\"");
                    break;
                case '\\':
                    output.append("\\\\");
                    break;
                case '\b':
                    output.append("\\b");
                    break;
                case '\f':
                    output.append("\\f");
                    break;
                case '\n':
                    output.append("\\n");
                    break;
                case '\r':
                    output.append("\\r");
                    break;
                case '\t':
                    output.append("\\t");
                    break;
                default:
                    if (character < 0x20) {
                        output.append(String.format(Locale.ROOT, "\\u%04x", (int) character));
                    } else {
                        output.append(character);
                    }
            }
        }
        output.append('"');
        return output.toString();
    }

    private static String jsonStringList(List<String> values) {
        if (values == null || values.isEmpty()) {
            return "null";
        }
        List<String> encoded = new ArrayList<String>();
        for (String value : values) {
            encoded.add(jsonString(value));
        }
        return "[" + String.join(",", encoded) + "]";
    }

    private static String blockJson(Block block) {
        return "{"
                + "\"block_type\":" + jsonString(block.type) + ","
                + "\"text\":" + jsonString(block.text) + ","
                + "\"page\":null,"
                + "\"section_path\":" + jsonStringList(block.sectionPath) + ","
                + "\"table_id\":" + jsonString(block.tableId) + ","
                + "\"row\":" + (block.row == null ? "null" : block.row.toString()) + ","
                + "\"column\":" + (block.column == null ? "null" : block.column.toString())
                + "}";
    }

    private static void writePayload(
            Path output,
            List<Block> blocks,
            String profile,
            String library) throws IOException {
        List<String> encoded = new ArrayList<String>();
        for (Block block : blocks) {
            // Preserve an empty table parent when the source table is a
            // layout-only grid.  The Python canonicalizer requires exactly
            // one parent for every emitted table_id, even when every cell is
            // blank.
            if (!block.text.isEmpty()
                    || "table".equals(block.type)
                    || "table_cell".equals(block.type)) {
                encoded.add(blockJson(block));
            }
        }
        String payload = "{"
                + "\"blocks\":[" + String.join(",", encoded) + "],"
                + "\"metadata\":{"
                + "\"worker\":\"java-hwp\","
                + "\"profile\":" + jsonString(profile) + ","
                + "\"library\":" + jsonString(library)
                + "},"
                + "\"raw_artifacts\":[]"
                + "}";
        Files.createDirectories(output.getParent());
        Files.write(output, payload.getBytes(StandardCharsets.UTF_8));
    }
}
