require 'csv'
require 'set'

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

      unique_values[column] ||= Set.new
      unique_values[column] << value
    end
  end
end

csv = CSV.open('infants_columns.csv', 'wb')

unique_values.sort_by do |k, values|
  values.size
end.each do |k, values|
  csv << [k, values.to_a]
end