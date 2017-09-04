require 'csv'
require 'json'

filenames = [
  "linkco2007us_den.csv",
  "linkco2008us_den.csv",
  "linkco2009us_den.csv",
  "linkco2010us_den.csv",
]

births_columns_counts = {}

filenames.each do |filename|
  index = -1
  columns = []

  lines_count = `wc -l #{filename}`.to_i
  puts "Processing [filename=#{filename},lines_count=#{lines_count}]"

  CSV.foreach(filename) do |values|
    index += 1

    if index == 0
      columns = values
      next
    end

    if index % 100000 == 0
      puts "Processing [filename=#{filename},index=#{index}]"
    end

    values.each_with_index do |value, index|
      column = columns[index]

      births_columns_counts[column] ||= {}
      births_columns_counts[column][value] ||= 0
      births_columns_counts[column][value] += 1
    end
  end

  puts "Processed [filename=#{filename}]"
end

csv_out = CSV.open('births_columns_counts.csv', 'wb')

births_columns_counts = births_columns_counts.sort_by do |column_name, columns_counts|
  columns_counts.size
end

births_columns_counts.each do |column_name, columns_counts|
  csv_out << [column_name, columns_counts.to_json]
end

csv_out = CSV.open('births_columns_values.csv', 'wb')

births_columns_counts.each do |column_name, columns_counts|
  csv_out << [column_name, columns_counts.values.to_json]
end
