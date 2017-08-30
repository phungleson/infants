require 'csv'
require 'json'

filenames = [
  "linkco2007us_den.csv",
  "linkco2008us_den.csv",
  "linkco2009us_den.csv",
  "linkco2010us_den.csv",
]

unique_values = {}

filenames.each do |filename|
  index = -1
  columns = []

  CSV.foreach(filename) do |values|
    index += 1

    if index == 0
      columns = values
      next
    end

    values.each_with_index do |value, index|
      column = columns[index]

      unique_values[column] ||= {}
      unique_values[column][value] ||= 0
      unique_values[column][value] += 1
    end

  end
end

csv = CSV.open('infants_columns_hash.csv', 'wb')

unique_values.sort_by do |k, values|
  values.size
end.each do |k, values|
  csv << [k, values.to_json]
end