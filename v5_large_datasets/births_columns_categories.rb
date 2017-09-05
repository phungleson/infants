require 'csv'
require 'json'
require 'finishing_moves'

births_columns_categories = {}

CSV.foreach('births_columns_counts.csv') do |row|
  column_name, values_hash = row
  values_hash = JSON.parse(values_hash)
  unique_values = values_hash.keys

  values = unique_values.select { |v| v != '' }

  if values.size > 0 && !values[0].numeric?
    # add a '' value in front so that we can signify missing values as 0
    values = values.unshift('')

    values.each_with_index do |value, index|
      value_number = index.to_f / (values.size - 1)

      births_columns_categories[column_name] ||= {}
      births_columns_categories[column_name][value] = value_number
    end
  end
end

csv_out = CSV.open('births_columns_categories.csv', 'wb')

births_columns_categories.each do |column_name, column_categories|
  csv_out << [column_name, column_categories.to_json]
end